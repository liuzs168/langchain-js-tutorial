import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';

import 'dotenv/config'; // pnpm add dotenv
import { AgentExecutor, createOpenAIToolsAgent } from '@langchain/classic/agents';
// 改为从新包导入
import { TavilySearch } from '@langchain/tavily';
import * as readline from 'node:readline/promises';
import * as process from 'node:process';
import { HumanMessage } from 'langchain';

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/classic/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";


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

const vectorStore = await createVectorStore()

/**检索数据 */
const retriever = vectorStore.asRetriever({
    k: 2,
})




// 创建model
const model = new ChatOpenAI({
    apiKey: process.env.ZHIPU_API_KEY,
    baseUrl: 'https://open.bigmodel.cn/api/paas/v4',
    model: 'glm-5.3-flash', // 替代已废弃的 modelName
    temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromMessages([
    ['system', "你是一个叫小悟的有用（乐于助人）的助手"],
    new MessagesPlaceholder("chat_history"),
    ['human', '{input}'],
    new MessagesPlaceholder("agent_scratchpad")
])

// 创建代理和工具
import { tool } from '@langchain/core/tools';
import { createRetrieverTool } from '@langchain/classic/tools/retriever';

const rawSearch = new TavilySearch({ maxResults: 3 });
const searchTool = tool(
    async (input) => JSON.stringify(await rawSearch.invoke(input)),
    { name: rawSearch.name, description: rawSearch.description, schema: rawSearch.schema }
);

const retrieverTool = createRetrieverTool(retriever, {
    name: 'Nobel_prize_search',
    description: '当搜索诺贝尔奖时使用此工具'
})
const tools = [searchTool, retrieverTool]; // ⚠️ 必须把工具放进去

const chatHistory = []
// 创建代理
const agent = await createOpenAIToolsAgent({
    llm: model,
    prompt,
    tools
})

// 创建代理执行器
const agentExecutor = new AgentExecutor({
    agent,
    tools
})


// 获取用户输入
const rl = readline.createInterface({ input: process.stdin, output: process.stdout })
const askQuestion = async () => {

    const input = await rl.question("User:")
    if (input.toLocaleLowerCase === "exit") {
        rl.close()
        return
    }
    const res = await agentExecutor.invoke({
        input,
        chat_history: chatHistory
    })
    console.log('Agent:', res.output)
    chatHistory.push(new HumanMessage(input))
    chatHistory.push(new HumanMessage(res.output))

    askQuestion()
}
askQuestion()


