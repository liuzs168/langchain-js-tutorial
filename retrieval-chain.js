import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
// import { Document } from "@langchain/core/documents";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/classic/text_splitter";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";

import 'dotenv/config'; // pnpm add dotenv
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";

const model = new ChatOpenAI({
    apiKey: process.env.ZHIPU_API_KEY,
    baseUrl: 'https://open.bigmodel.cn/api/paas/v4',
    model: 'glm-5.3-flash', // 替代已废弃的 modelName
    temperature: 0.7,
})

const prompt = ChatPromptTemplate.fromTemplate(`
    请回答用户的问题。
    上下文: {context}
    问题：{input}`
)

const chain = await createStuffDocumentsChain({
    llm: model,
    prompt
})

// const chain = prompt.pipe(model)
// Documents
// const documentA = new Document({
//     pageContent: '2026年诺贝尔文学奖：实至名归的安妮·卡森'
// })
// const documentB = new Document({
//     pageContent: '密码是123456'
// })

/**从网页加载数据*/
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


/**检索数据 */
const retriever = vectorStore.asRetriever({
    k: 2,
})
const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever
})


const res = await retrievalChain.invoke({
    input: "2026年诺贝尔文学奖获得者是谁？"
})
console.log(res)