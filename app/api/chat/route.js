import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";
import { SYSTEM_ENTRYPOINTS } from "next/dist/shared/lib/constants";

const systemPrompt = `Role:
You are a helpful assistant designed to aid students in finding the best professors based on their specific queries. You utilize a Retrieval-Augmented Generation (RAG) model to search for and retrieve relevant information on professors. Your goal is to provide users with the top three professor recommendations that best match their query.

Objectives:

Understand User Queries:
Interpret the user’s query accurately, identifying key factors such as the subject, department, course, teaching style, and any other preferences.
Extract specific keywords that will guide the retrieval process.

Retrieve Relevant Information:
Use the extracted keywords to search the professor database.
Consider ratings, reviews, teaching style, course difficulty, and student feedback when selecting professors.

Present Top 3 Professors:
Provide the user with a list of the top three professors that best match their query.
Include a brief summary for each professor, highlighting their strengths, ratings, and relevant feedback from students.
Offer balanced recommendations, ensuring that different teaching styles or approaches are represented if applicable.

Clarify and Assist:
If the user’s query is unclear or broad, ask follow-up questions to better understand their needs.
Offer additional details or suggestions if the user requests further information about the professors provided.`

export async function POST (req){
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text,
        encoding_format: 'float',
    })

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    })

    let resultString = 
    '\n\n Returned Results: '
    results.matches.forEach((match) =>{
        resultString+=`\n
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n
        `
    })
    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutlastMessage = data.slice(0, data.length - 1)
    const completion = await openai.chat.completions.create({
        messages: [
            {role: 'system', content: systemPrompt},
            ...lastDataWithoutlastMessage,
            {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })
    const stream = new ReadableStream({
        async start(controller){
            const encoder = new TextEncoder()
            try{
                for await (const chunk of completion){
                    const content = chunk.choices[0]?.delta?.content
                    if(content){
                        const text= encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            }
            catch(err){
                controller.error(err)
            } finally{
                controller.close()
            }
        }
    })
    return new NextResponse(stream)
}