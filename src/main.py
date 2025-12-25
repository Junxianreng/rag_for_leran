import sys
import os
import shutil
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from core_2 import rag_service  # 导入核心服务

app = FastAPI(title="zhipu RAG API")

# ========== 请求体模型 ==========
# 删除文档的请求体
class DeleteDocRequest(BaseModel):
    doc_id: str

# 流式问答的请求体
class ChatStreamRequest(BaseModel):
    question: str
    model_name: str = "glm-4.5-air"

# ========== 核心接口==========
@app.get("/")
def read_root():
    return {"status": "RAG Backend is running"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件接口"""
    temp_dir = "temp_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    file_location = f"{temp_dir}/{file.filename}"
    # 保存上传的文件
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 调用核心服务处理
        num_chunks = rag_service.upload_document(file_location)
        os.remove(file_location)  # 清理临时文件
        return {"filename": file.filename, "chunks_added": num_chunks, "message": "Index built successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/list_documents")
async def list_documents():
    """获取已上传文档列表接口"""
    try:
        doc_list = rag_service.list_documents()
        return doc_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败：{str(e)}")

@app.post("/api/delete_document")
async def delete_document(req: DeleteDocRequest):
    """删除指定doc_id的文档接口"""
    try:
        success = rag_service.delete_document(req.doc_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"未找到doc_id={req.doc_id}的文档")
        return {"success": success, "message": "文档删除成功"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败：{str(e)}")

@app.post("/api/chat_stream")
async def chat_stream(req: ChatStreamRequest):
    """流式问答接口（前端核心使用）"""
    def generate_response():
        try:
            for chunk in rag_service.chat_stream(req.question, req.model_name):
                if "参考来源" in chunk:
                    sources = chunk.split("参考来源：")[1].strip().split("\n")
                    yield json.dumps({"sources": sources}) + "\n"
                else:
                    yield json.dumps({"content": chunk}) + "\n"
        except Exception as e:
            yield json.dumps({"content": f"问答出错：{str(e)}"}) + "\n"

    return StreamingResponse(generate_response(), media_type="application/json")

# ========== 启动服务 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
