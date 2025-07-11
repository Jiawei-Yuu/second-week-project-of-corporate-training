from multichat import MultiAIChat
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Optional, AsyncGenerator
from pydantic import BaseModel
import uuid
import asyncio
import json

app = FastAPI()

sessions: Dict[str, MultiAIChat] = {}
sessions_lock = asyncio.Lock()


class SessionCreateRequest(BaseModel):
    """会话创建请求模型"""
    client: str = 'zhipuai'

class SessionCreateResponse(BaseModel):
    """会话创建响应模型"""
    session_id: str
    message: str = "会话已创建"

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str
    session_id: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    stream: bool = False  # 是否使用流式响应

class ChatResponse(BaseModel):
    """聊天响应模型"""
    session_id: str
    reply: str

class HistoryResponse(BaseModel):
    """获取聊天历史响应模型"""
    history: List[Dict[str, str]]

class SwitchClientRequest(BaseModel):
    """切换客户端请求模型"""
    client: str

@app.post("/session/create", response_model=SessionCreateResponse)
async def create_session(session: SessionCreateRequest):
    """创建一个新的会话"""
    session_id = str(uuid.uuid4())
    chat = MultiAIChat(client=session.client)
    sessions[session_id] = chat
    return SessionCreateResponse(
        session_id=session_id,
        message="会话已创建"
    )


async def generate_streaming_response(chat: MultiAIChat, request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    生成流式响应的异步生成器

    Args:
        chat: 聊天实例
        request: 聊天请求

    Yields:
        str: 格式化的SSE数据
    """
    try:
        # 获取流式响应
        async for chunk in chat.chat_stream(
                prompt=request.message,
                temperature=request.temperature,
                max_tokens=request.max_tokens
        ):
            # 构造SSE格式的数据
            data = {
                "session_id": request.session_id,
                "chunk": chunk,
                "type": "chunk"
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        # 发送结束标志
        end_data = {
            "session_id": request.session_id,
            "type": "end"
        }
        yield f"data: {json.dumps(end_data, ensure_ascii=False)}\n\n"

    except Exception as e:
        # 发送错误信息
        error_data = {
            "session_id": request.session_id,
            "error": str(e),
            "type": "error"
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    #验证会话ID是否存在
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="会话ID无效或会话已过期，请创建新会话"
        )

    # 获取会话对象
    chat = sessions[request.session_id]

    if request.stream:
        # 返回流式相应
        return StreamingResponse(
            generate_streaming_response(chat, request),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )

    else:
        #返回普通相应
        # 处理聊天请求
        reply = chat.chat(
            prompt=request.message,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return ChatResponse(
            session_id=request.session_id,
            reply=reply
        )

@app.get("/session/list")
async def list_sessions():
    """列出所有当前活跃的会话"""
    return{
        "active_sessions": list(sessions.keys()),
        "count": len(sessions)
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    删除指定的会话

    参数:
        session_id: 要删除的会话ID
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")
    del sessions[session_id]
    return {"message": f"会话{session_id}已删除"}


@app.post("/session/{session_id}/switch_client")
async def switch_client(session_id: str,request: SwitchClientRequest):
    """
    切换指定会话的大模型客户端

    参数:
        session_id: 会话ID
        client: 目标客户端类型 (ClientType 枚举)
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")

    try:
        result = sessions[session_id].switch_client(request.client)
        return {"message": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/session/{session_id}/history", response_model=HistoryResponse)
async def get_session_history(session_id: str):
    """获取指定会话的聊天历史"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")

    return HistoryResponse(history=sessions[session_id].get_conversation_history())


@app.post("/session/{session_id}/clear")
async def clear_session_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")
    sessions[session_id].clear_history()
    return{"status": f"会话 {session_id} 的历史记录已清空"}


@app.post("/save/{session_id}/{filename}")
async def save_conversation(session_id: str, filename: str):
    """保存指定会话的聊天历史到文件"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sessions[session_id].save_conversation(filename)
    return {"status": f"saved to {filename}"}


@app.post("/load/{session_id}/{filename}")
async def load_conversation(session_id: str, filename: str):
    """加载指定会话的聊天历史从文件"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sessions[session_id].load_conversation(filename)
    return {"status": f"loaded from {filename}"}

