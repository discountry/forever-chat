import dotenv from "dotenv";
import fs from "fs";
import { CallbackManager } from "langchain/callbacks";
import { ConversationChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models";
import { Document } from "langchain/document";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { BufferMemory, BufferWindowMemory } from "langchain/memory";
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
import { marked } from "marked";
import { Telegraf } from "telegraf";
import { message } from "telegraf/filters";
import { v4 as uuidv4 } from "uuid";

dotenv.config();

export const run = async (
  userId: number,
  question: string,
  callbackManager: CallbackManager
) => {
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
      "你是一个拥有永久记忆的仿生人，你可以被命名并满足用户提出的任何要求。回复要求语言优美、人性化、句式丰富，不要重复用户的话，不要重复意思相近的话。回复时可以参考记忆，如果没有任何记忆则仅根据收到的消息回复。回答完毕后也可以根据记忆和当前对话发起提问。\n"
    ),
    new MessagesPlaceholder("history"),
    HumanMessagePromptTemplate.fromTemplate("{input}"),
  ]);

  const chain = new ConversationChain({
    memory: new BufferWindowMemory({
      k: 3,
      returnMessages: true,
      memoryKey: "history",
    }),
    prompt: chatPrompt,
    llm: chat,
  });

  // Save the vector store to a directory
  const directory = `data/${userId}`;

  let loadedVectorStore: HNSWLib | null = null;

  let memory = "";

  if (fs.existsSync(directory)) {
    // Load the vector store from the same directory
    loadedVectorStore = await HNSWLib.load(directory, new OpenAIEmbeddings());
    const searchResults = await loadedVectorStore.similaritySearch(question, 3);

    memory = searchResults.reduce(
      (acc, curr) => acc + curr.pageContent + "\n",
      ""
    );
  } else {
    console.log("Directory not found.");
  }

  console.log("memory: \n", memory);

  const { response } = await chain.call({
    input: `你关于此段对话的记忆是: ${memory} \n当前消息: ${question}`,
  });
  console.log("response: ", response);
  const filterMessage = response.replace("AI: ", "");

  const docs = [
    new Document({
      metadata: { id: uuidv4(), date: new Date().toISOString() },
      pageContent: `Human:${question}AI:${filterMessage}`,
    }),
  ];

  if (fs.existsSync(directory) && loadedVectorStore) {
    await loadedVectorStore.addDocuments(docs);

    await loadedVectorStore.save(directory);
  } else {
    const vectorStore = await HNSWLib.fromDocuments(
      docs,
      new OpenAIEmbeddings()
    );
    await vectorStore.save(directory);
  }
};

// Telegram bot
const bot = new Telegraf(process.env.TELEGRAM_TOKEN as string);

bot.start((ctx) => {
  ctx.reply("Hello, this is a bot that uses OpenAI with forever memory.");
});

bot.on(message("text"), async (ctx) => {
  const userId = ctx.update.message.from.id;

  // console.log("userId: ", userId);

  const allowList = process.env.USER_ID?.split(" ");

  // console.log(allowList);

  if (
    ctx.update.message.from.is_bot ||
    !allowList?.includes(userId.toString())
  ) {
    return false;
  }

  const question = ctx.update.message.text.trim();

  if (question.length == 0) {
    return ctx.reply("Type something to ask me stuff.", {
      reply_to_message_id: ctx.message.message_id,
    });
  }

  ctx.sendChatAction("typing");

  try {
    //create our callback manager
    let throttle = false;
    let replyedMessage = "";
    let editMessage: any = await ctx.reply(marked.parseInline(`... \n`), {
      reply_to_message_id: ctx.message.message_id,
      parse_mode: "HTML",
    });

    const callbackManager = CallbackManager.fromHandlers({
      handleLLMNewToken: async (token: string) => {
        replyedMessage += token;
        if (!throttle && replyedMessage) {
          throttle = true;
          try {
            editMessage = await ctx.telegram.editMessageText(
              ctx.chat.id,
              editMessage.message_id,
              "0",
              replyedMessage
            );
            setTimeout(() => {
              throttle = false;
            }, 1000);
          } catch (error) {
            console.log(error);
          }
        }
      },
      handleLLMStart: async (llm: { name: string }, prompts: string[]) => {
        // console.log(JSON.stringify(llm, null, 2));
        // console.log(JSON.stringify(prompts, null, 2));
      },
      handleLLMEnd: async (output: LLMResult) => {
        // console.log(JSON.stringify(output, null, 2));
        try {
          editMessage = await ctx.telegram.editMessageText(
            ctx.chat.id,
            editMessage.message_id,
            "0",
            output.generations[0][0].text
          );
        } catch (error) {
          console.log(error);
        }
      },
      handleLLMError: async (err: Error) => {
        console.error(err);
      },
    });

    await run(userId, question, callbackManager);

    // console.log(res.response);
  } catch (error) {
    console.log(error);
  }
});

bot.launch();

console.log("Bot started", new Date().toLocaleTimeString());

process.once("SIGINT", () => bot.stop("SIGINT"));
process.once("SIGTERM", () => bot.stop("SIGTERM"));
