
import * as dotenv from 'dotenv';
dotenv.config();

import readlineSync from 'readline-sync';

import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";


const ai = new GoogleGenAI({});
const History = []

async function transformQuery(question) {

    History.push({
        role: 'user',
        parts: [{ text: question }]
    })

    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: History,
        config: {
            systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
    Only output the rewritten question and nothing else.
      `,
        },
    });

    History.pop()

    return response.text


}

async function chatting(question){

    const goodquery = await transformQuery(question);
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'text-embedding-004',
    });
    const queryVector = await embeddings.embedQuery(goodquery);
    

    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    const searchResults = await pineconeIndex.query({
        topK: 10,
        vector: queryVector,
        includeMetadata: true,
    });

    //console.log(searchResults);
    const context = searchResults.matches
        .map(match => match.metadata.text)
        .join("\n\n---\n\n");


    History.push({
        role: 'user',
        parts: [{ text: goodquery }]
    })


    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: History,
        config: {
            systemInstruction: `You are an expert teacher specializing in Data Structures, Algorithms, and Competitive Programming.
You will be given a context from the CSESBook and a user question. If the answer is present, explain it in a clear, concise, and educational way.If the information is not available in the context, respond by politely stating that the answer cannot be found in the provided document (you may phrase this naturally, not restricted to a fixed wording). 
      
      Context: ${context}
      `,
        },
    });


    History.push({
        role: 'model',
        parts: [{ text: response.text }]
    })

    console.log("\n");
    console.log(response.text);
}


async function main() {
    const userProblem = readlineSync.question("Ask me anything--> ");
    await chatting(userProblem);
    main();
}


main();