import dotenv from "dotenv";
import {
  createVectorStoreAgent,
  VectorStoreInfo,
  VectorStoreToolkit,
} from "langchain/agents";
import { CallbackManager } from "langchain/callbacks";
import { ChatVectorDBQAChain, ConversationChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models";
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
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
    cache: true,
    concurrency: 5,
    verbose: true,
    streaming: true,
    callbackManager,
  },
  {
    basePath: "https://openai.yubolun.com/v1",
  }
);

// Save the vector store to a directory
const directory = "data";
//   await vectorStore.save(directory);

// Load the vector store from the same directory
const loadedVectorStore = await HNSWLib.load(directory, new OpenAIEmbeddings());

const template = "What is a good name for a company that makes {product}?";
const prompt = new PromptTemplate({
  template: template,
  inputVariables: ["product"],
});
/* Create the chain */
const chain = ChatVectorDBQAChain.fromLLM(chat, loadedVectorStore, {
  k: 1,
});

/* Ask it a question */
// const question = "Do you know pinecone?";
// const res = await chain.call({
//   question,
//   chat_history: [],
// });

// console.log(res);

// const chatHistory = question + res["text"];
// const followUpRes = await chain.call({
//   question: "Do you like it?",
//   chat_history: chatHistory,
// });
// console.log(followUpRes);
