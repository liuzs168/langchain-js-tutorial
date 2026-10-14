import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { CommaSeparatedListOutputParser } from '@langchain/core/output_parsers';
import { StructuredOutputParser } from '@langchain/core/output_parsers';
import { z } from 'zod';

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
const prompt = ChatPromptTemplate.fromMessages([
    ['system', '你是一个喜剧演员，你根据一个词语生成一个笑话'],
    ['human', '{input}'],
]);

async function callStringOutputParser() {

    const parser = new StringOutputParser();

    //创建chain
    const chain = prompt.pipe(model).pipe(parser);

    //调用chain
    return await chain.invoke({ input: '狗' });
}


async function callListOutputParser() {
    const prompt = ChatPromptTemplate.fromTemplate('生成词语{input}的五个同义词，用逗号分隔');
    const outputParser = new CommaSeparatedListOutputParser();

    const chain = prompt.pipe(model).pipe(outputParser);
    return await chain.invoke({ input: '开心' });
}

// structured output parser
async function callStructuredParser() {
    const prompt = ChatPromptTemplate.fromTemplate(`
        从以下短语中提取信息。
        格式说明：{format_instructions}
        短语：{phrase}
    `);

    const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
        name: '这个人的名字',
        age: '这个人的年龄',
    });

    const chain = prompt.pipe(model).pipe(outputParser);
    return await chain.invoke({
        phrase: '这个人的名字是张三，他的年龄是25岁。',
        format_instructions: outputParser.getFormatInstructions(),
    });

}

async function callZodOutputParser() {
    const prompt = ChatPromptTemplate.fromTemplate(`
        从以下短语中提取信息。
        格式说明：{format_instructions}
        短语：{phrase}
    `);

    const outputParser = StructuredOutputParser.fromZodSchema(
        z.object({
            recipe: z.string().describe('食谱名称'),
            ingredients: z.array(z.string()).describe('食材列表'),
        })
    );

    const chain = prompt.pipe(model).pipe(outputParser);
    return await chain.invoke({
        phrase: '食谱名称是番茄炒蛋，食材有番茄、鸡蛋、盐、油。',
        format_instructions: outputParser.getFormatInstructions(),
    });

}


// const res = await callStringOutputParser();
// const res = await callListOutputParser();
// const res = await callStructuredParser();
const res = await callZodOutputParser();
console.log(res);