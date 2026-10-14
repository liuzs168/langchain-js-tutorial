import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';

import 'dotenv/config'; // pnpm add dotenv


// 创建model
const model = new ChatOpenAI({
    apiKey: process.env.ZHIPU_API_KEY,
    baseUrl: 'https://open.bigmodel.cn/api/paas/v4',
    model: 'glm-5.3-flash', // 替代已废弃的 modelName
    temperature: 0.7,
    maxTokens: 1000,
    // verbose: true, // 开启调试模式
});

// 创建prompt模板
const prompt = ChatPromptTemplate.fromTemplate('你是一个喜剧演员，你根据词语{input}生成一个笑话');


//创建chain
const chain = prompt.pipe(model);

//调用chain
const res = await chain.invoke({ input: '狗' });
console.log(res.content);