import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
// import { Document } from "@langchain/core/documents";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/classic/text_splitter";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RunnablePassthrough } from "@langchain/core/runnables";

import 'dotenv/config'; // pnpm add dotenv
import { AIMessage, HumanMessage } from "langchain";
import { createHistoryAwareRetriever } from "@langchain/classic/chains/history_aware_retriever";

// const chain = prompt.pipe(model)
// Documents
// const documentA = new Document({
//     pageContent: '2026年诺贝尔文学奖：实至名归的安妮·卡森'
// })
// const documentB = new Document({
//     pageContent: '密码是123456'
// })

// 对话历史（假的历史纪录）
const chatHistory = [
    new HumanMessage('你好！'),
    new AIMessage('我能帮你什么？'),
    new HumanMessage('我叫小新'),
    new AIMessage('小新你好，有什么能帮你的吗？'),
    new HumanMessage('2026年诺贝尔文学奖获得者是谁？'),
    new AIMessage('2026年诺贝尔文学奖获得者是加拿大诗人安妮·卡森（Anne Carson）')
]

/**从网页加载数据,并创建向量存储*/
const createVectorStore = async () => {
    const loader = new CheerioWebBaseLoader(
        "https://news.southcn.com/node_64549305f1/dd650c5a6d.shtml"
    )
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 20
    })
    const splitDocs = await splitter.splitDocuments(docs)


    const embeddings = new OpenAIEmbeddings({
        apiKey: process.env.ZHIPU_API_KEY,
        model: "embedding-3",
        batchSize: 64,
        configuration: {
            baseURL: "https://open.bigmodel.cn/api/paas/v4",
        },
    })
    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings)
    return vectorStore
}


/** 创建检索链 */
const createChain = async (vectorStore) => {

    const model = new ChatOpenAI({
        apiKey: process.env.ZHIPU_API_KEY,
        baseUrl: 'https://open.bigmodel.cn/api/paas/v4',
        model: 'glm-5.3-flash', // 替代已废弃的 modelName
        temperature: 0.7,
    })

    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "请基于上下文回答用户的问题。上下文: {context}"],
        new MessagesPlaceholder("chat_history"),
        ['human', "{input}"]
    ])


    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt
    })

    /**检索数据 */
    const retriever = vectorStore.asRetriever({
        k: 2,
    })

    // 重新表述提示模板
    const retrieverPrompt = ChatPromptTemplate.fromMessages([
        new MessagesPlaceholder("chat_history"),
        ['human', "{input}"],
        ['human', '请根据上述对话，提供另一条消息，并生成一个搜索查询，以查找对话相关的消息']
    ])
    const historyAwareRetriever = await createHistoryAwareRetriever(
        {
            llm: model,
            retriever,
            rephrasePrompt: retrieverPrompt
        }
    )

    const conversationChain = RunnablePassthrough.assign({
        context: (input) => historyAwareRetriever.invoke(input),
        chat_history: (input) => input.chat_history ?? []
    }).assign({
        answer: chain
    })

    return conversationChain

}


const vectorStore = await createVectorStore()
const chain = await createChain(vectorStore)



const res = await chain.invoke({
    input: "她是谁？",
    chat_history: chatHistory
})
console.log(res)