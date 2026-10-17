import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';

import 'dotenv/config'; // pnpm add dotenv
import { BufferMemory } from '@langchain/classic/memory';
import { ConversationChain } from '@langchain/classic/chains';
import { UpstashRedisChatMessageHistory } from '@langchain/community/stores/message/upstash_redis';


// 创建model
const model = new ChatOpenAI({
    apiKey: process.env.ZHIPU_API_KEY,
    baseUrl: 'https://open.bigmodel.cn/api/paas/v4',
    model: 'glm-5.3-flash', // 替代已废弃的 modelName
    temperature: 0.7,

});

// 创建prompt模板
const prompt = ChatPromptTemplate.fromTemplate(`
    你是一个AI助手。
    History: {history}
    {input}
`);

const upstashChatHistory = new UpstashRedisChatMessageHistory({
    sessionId: "chat1",
    config: {
        url: process.env.UPSTASH_REST_URL,
        token: process.env.UPSTASH_REST_TOKEN
    }
})

const memory = new BufferMemory({
    memoryKey: 'history',
    chatHistory: upstashChatHistory
})


//创建chain
const chain = new ConversationChain({
    llm: model,
    prompt,
    memory
})
// const chain = prompt.pipe(model);

//调用chain
console.log(await memory.loadMemoryVariables())
const input1 = {
    input: '密码是123456'
}
const res1 = await chain.invoke({ input: input1.input });
console.log(res1);


console.log('更新后的历史', await memory.loadMemoryVariables())
const input2 = {
    input: '密码是什么'
}
const res2 = await chain.invoke({ input: input2.input });
console.log(res2);

