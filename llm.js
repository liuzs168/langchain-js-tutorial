import { ChatOpenAI } from '@langchain/openai';
import 'dotenv/config'; // pnpm add dotenv

const model = new ChatOpenAI({
    apiKey: process.env.ZHIPU_API_KEY,
    baseUrl: 'https://open.bigmodel.cn/api/paas/v4',
    model: 'glm-5.3-flash', // 替代已废弃的 modelName
    temperature: 0.7,
    maxTokens: 1000,
    verbose: true, // 开启调试模式
});

console.log('模型加载完成');
const res = await model.invoke('你好');
console.log(res);


// for await (const chunk of res) {
//     console.log(chunk.content);
// }