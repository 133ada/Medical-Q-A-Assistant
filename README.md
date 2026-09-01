# 医疗问答助手后端

医疗问答助手（Medical Q&A Assistant）的后端服务，基于 **FastAPI + LangGraph + RAG + 向量数据库 + Redis/关系型数据库** 构建，面向医疗知识问答、会话管理、用户认证和对话记忆等场景。

> 本文档仅介绍 App 后端目录，不包含前端使用说明。

## 主要能力

- **医疗知识问答**：基于检索增强生成（RAG）从医疗知识库中检索相关内容，再交由医疗问答 Agent 生成回答。
- **药物相互作用检索**：单独维护 DrugBank DDI（药物相互作用）向量库。
- **LangGraph Agent 编排**：组织问题分类、知识检索、上下文处理和回答生成流程。
- **流式回答**：返回 Agent 执行进度和最终回答，适合聊天界面实时展示。
- **用户认证**：支持注册、登录和 JWT Bearer Token 认证。
- **会话管理**：登录用户可以创建会话、查看最近会话、查询历史消息并继续对话。
- **分层记忆**：Redis 或内存保存短期消息；关系型数据库保存会话、摘要和长期记忆；上下文达到 Token 阈值后自动生成摘要，并按相关性召回长期记忆。
- **健康检查与图可视化**：提供存活检查、就绪检查和 LangGraph Mermaid 图接口。
- **匿名访问**：默认允许匿名提问；匿名请求不写入用户会话和长期记忆。

## 技术栈

- Python 3.10+
- FastAPI / Uvicorn
- Pydantic
- LangChain / LangGraph
- ChromaDB
- SQLAlchemy
- PostgreSQL 或 SQLite
- Redis（不可用时可降级到进程内存）
- JWT / bcrypt 密码哈希
- DeepSeek：问答模型
- 阿里云 DashScope：文本 Embedding 模型

## 后端目录结构

~~~text
App/
├── main.py                    # FastAPI 应用入口、生命周期和健康检查
├── Core/                      # 配置、数据库门面、依赖注入和异常定义
├── Routers/                   # 推荐使用的 API 路由
├── Chat/                      # 兼容旧版本的聊天路由实现
├── Services/                  # 认证、聊天、RAG、记忆等业务编排
├── Integrations/              # LangGraph、LLM、RAG、Memory 外部集成
├── LangGraph/                 # 医疗 Agent 图及实验性图编排代码
├── LLM/                       # 大模型和 Embedding 模型封装
├── Rag/                       # Chroma 向量库及增量构建逻辑
├── Memory/                    # 短期记忆、摘要和长期记忆基础设施
├── Models/                    # SQLAlchemy 数据模型
├── Repositories/              # 用户、会话、消息和记忆的数据访问层
├── Schemas/                   # Pydantic 请求/响应模型
├── Security/                  # JWT、密码哈希和认证逻辑
├── Skills/                    # 医疗技能和 Agent 技能注册表
└── Utils/                     # 通用认证等工具
~~~

## API 接口

应用默认监听 http://127.0.0.1:8000。

### 基础接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | / | 返回服务状态和主要接口提示 |
| GET | /health/live | 存活检查 |
| GET | /health/ready | 检查 Agent、向量库、数据库和短期记忆依赖是否就绪 |
| GET | /graph/viz | 返回 LangGraph Mermaid 图 |
| GET | /docs | FastAPI Swagger 文档 |
| GET | /redoc | FastAPI ReDoc 文档 |

### 认证接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /api/auth/register | 注册用户并返回 JWT |
| POST | /api/auth/login | 用户登录并返回 JWT |
| GET | /api/auth/me | 获取当前登录用户，需要 Bearer Token |

注册请求示例：

~~~json
{
  "username": "demo_user",
  "password": "your-password",
  "nickname": "演示用户"
}
~~~

### 问答接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /api/chat | 医疗问答；匿名请求不持久化，登录请求保存会话和消息 |
| POST | /api/chat/stream | 流式返回检索和回答过程 |
| POST | /chat | 旧版兼容接口，已标记为 deprecated |
| POST | /chat/stream | 旧版流式兼容接口，已标记为 deprecated |

问答请求示例：

~~~json
{
  "question": "阿司匹林和华法林同时使用需要注意什么？",
  "session_id": null,
  "conversation_id": "optional-conversation-id"
}
~~~

### 会话接口

以下接口均需要登录：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | /api/sessions | 查询最近 7 天的会话 |
| POST | /api/sessions | 创建会话 |
| GET | /api/sessions/{session_id} | 查询会话详情及历史消息 |
| POST | /api/sessions/{session_id}/messages | 在指定会话中追加问题 |

认证请求需要携带：

~~~http
Authorization: Bearer <JWT_TOKEN>
~~~

## 请求处理流程

1. FastAPI 启动并读取环境变量。
2. 初始化 DeepSeek 问答模型和阿里云 Embedding 模型。
3. 加载或增量更新普通医疗知识库、药物相互作用知识库。
4. 初始化数据库和短期记忆存储。
5. 构建 LangGraph 医疗 Agent。
6. 接收问题并解析登录状态。
7. 检索医疗知识、药物信息及历史记忆。
8. 生成普通或流式回答。
9. 对登录用户持久化会话和消息，并在后台更新短期记忆、摘要和长期记忆。

## 配置说明

后端会读取项目根目录的 .env，也兼容读取 App/.env。请根据项目根目录的 .env.example 创建本地配置文件。

**不要将真实的 .env、API Key、数据库密码或 JWT Secret 提交到代码仓库。**

核心配置项：

| 变量 | 说明 |
|---|---|
| DEEPSEEK_API_KEY | DeepSeek 问答模型 API Key |
| DEEPSEEK_BASE_URL | DeepSeek API 地址 |
| DEEPSEEK_MODEL | DeepSeek 模型名称 |
| ALIYUN_API_KEY | 阿里云 DashScope API Key |
| ALIYUN_BASE_URL | Embedding 服务地址 |
| ALIYUN_EMBEDDING_MODEL | Embedding 模型名称，默认 text-embedding-v3 |
| JWT_SECRET | JWT 签名密钥，生产环境必须使用高强度随机值 |
| DATABASE_URL | 数据库连接串；生产环境建议 PostgreSQL |
| REDIS_URL | Redis 连接串，默认 redis://127.0.0.1:6379/0 |
| AUTH_REQUIRED | 是否强制所有业务请求认证 |
| AUTH_TOKENS_JSON | 兼容旧 Token 映射配置 |
| FRONTEND_ORIGINS | CORS 允许的前端来源，多个地址用逗号分隔 |
| CHAT_REPLY_MODE | 聊天回答模式配置 |

默认运行时至少需要配置 DEEPSEEK_API_KEY、ALIYUN_API_KEY 和 JWT_SECRET。

## 本地运行

在项目根目录执行：

~~~powershell
# 安装后端依赖（依赖清单以项目实际配置为准）
pip install -r requirements.txt

# 启动后端
uvicorn App.main:app --host 127.0.0.1 --port 8000 --reload
~~~

也可以直接运行入口文件：

~~~powershell
python App/main.py
~~~

启动后可访问：

- Swagger： http://127.0.0.1:8000/docs
- ReDoc： http://127.0.0.1:8000/redoc
- 存活检查： http://127.0.0.1:8000/health/live

## 数据与向量库

- 普通医疗知识向量库位于 App/Rag/chroma_drugbank_other/。
- 药物相互作用向量库位于 App/Rag/chroma_drugbank_ddi/。
- 向量库使用 ChromaDB 持久化，并通过文件指纹实现增量处理。
- 数据库用于保存用户、会话、消息、会话摘要和长期记忆。
- Redis 主要用于短期对话记忆；Redis 不可用时可根据配置降级为内存存储。

向量库和本地数据库可能包含较大的二进制或运行时数据。发布源码时应根据仓库策略决定是否提交；如果仓库仅用于代码协作，建议单独提供数据初始化或导入方案。

## 安全注意事项

1. 严禁提交真实的 .env 文件和任何密钥。
2. 生产环境使用随机、高强度的 JWT_SECRET。
3. 将 FRONTEND_ORIGINS 限制为实际前端域名。
4. 生产环境使用 PostgreSQL、Redis 等受控服务，并限制网络访问权限。
5. 医疗问答结果仅供健康信息参考，不能替代医生诊断、处方或急救建议。
6. 对外发布 API 时建议增加 HTTPS、访问频率限制、日志脱敏和权限审计。

## 免责声明

本项目用于技术研究和个人学习。系统生成的内容可能存在遗漏或错误，不能替代专业医疗机构和执业医师的诊断与治疗意见。遇到急症或严重症状，请及时联系当地急救服务或前往正规医疗机构就诊。
