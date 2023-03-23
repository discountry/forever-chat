import dotenv from "dotenv";
import { CallbackManager } from "langchain/callbacks";
import { ConversationChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models";
import { Document } from "langchain/document";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { BufferMemory } from "langchain/memory";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
  PromptTemplate,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import {
  AIChatMessage,
  HumanChatMessage,
  LLMResult,
  SystemChatMessage,
} from "langchain/schema";
import { HNSWLib } from "langchain/vectorstores";
import { v4 as uuidv4 } from "uuid";

dotenv.config();

//create our callback manager

const callbackManager = CallbackManager.fromHandlers({
  handleLLMNewToken: async (token: string) => {
    // console.log(token);
  },
  handleLLMStart: async (llm: { name: string }, prompts: string[]) => {
    console.log(JSON.stringify(llm, null, 2));
    console.log(JSON.stringify(prompts, null, 2));
  },
  handleLLMEnd: async (output: LLMResult) => {
    console.log(JSON.stringify(output, null, 2));
  },
  handleLLMError: async (err: Error) => {
    console.error(err);
  },
});

const chat = new ChatOpenAI(
  {
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
    cache: true,
    concurrency: 5,
    verbose: true,
    streaming: true,
    callbackManager,
  },
  {
    basePath: process.env.BASE_PATH,
  }
);

const chatPrompt = ChatPromptTemplate.fromPromptMessages([
  SystemMessagePromptTemplate.fromTemplate(
    "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."
  ),
  new MessagesPlaceholder("history"),
  HumanMessagePromptTemplate.fromTemplate("{input}"),
]);

const chain = new ConversationChain({
  memory: new BufferMemory({ returnMessages: true, memoryKey: "history" }),
  prompt: chatPrompt,
  llm: chat,
});

// Save the vector store to a directory
const directory = "data";

// Load the vector store from the same directory
const loadedVectorStore = await HNSWLib.load(directory, new OpenAIEmbeddings());

const question = "What is pinecone?";

const searchResults = await loadedVectorStore.similaritySearch("pinecone", 1);
console.log(searchResults);

const memory = searchResults[0].pageContent;

const res = await chain.call({
  input: `Your memory of this conversation is: ${memory} Question: ${question}`,
});

console.log(res.response);

const docs = [
  new Document({
    metadata: { id: uuidv4() },
    pageContent: `Question: ${question} Answer: ${res.response}`,
  }),
];

// console.log(docs);

await loadedVectorStore.addDocuments(docs);

await loadedVectorStore.save(directory);
