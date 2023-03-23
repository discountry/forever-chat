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
import { marked } from "marked";
import { Telegraf } from "telegraf";
import { v4 as uuidv4 } from "uuid";

dotenv.config();

//create our callback manager
let currentMessage = "";
let replyedMessage = "";

const callbackManager = CallbackManager.fromHandlers({
  handleLLMNewToken: async (token: string) => {
    currentMessage += token;
    // console.log(currentMessage);
  },
  handleLLMStart: async (llm: { name: string }, prompts: string[]) => {
    // console.log(JSON.stringify(llm, null, 2));
    // console.log(JSON.stringify(prompts, null, 2));
  },
  handleLLMEnd: async (output: LLMResult) => {
    // console.log(JSON.stringify(output, null, 2));
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

// Telegram bot
const bot = new Telegraf(process.env.TELEGRAM_TOKEN as string);

bot.start((ctx) => {
  ctx.reply(
    "Hello, this is a bot that uses OpenAI.\nAsk anything using /ask followed by your question."
  );
});

bot.command("ask", async (ctx) => {
  currentMessage = "";
  replyedMessage = "";

  const userId = ctx.update.message.from.id;

  // console.log("userId: ", userId);

  if (
    ctx.update.message.from.is_bot ||
    userId !== Number(process.env.USER_ID)
  ) {
    return false;
  }

  const args = ctx.update.message.text.split(" ");
  args.shift();
  let question = args.join(" ");

  if (question.length == 0) {
    return ctx.reply("Type something after /ask to ask me stuff.", {
      reply_to_message_id: ctx.message.message_id,
    });
  }

  ctx.sendChatAction("typing");

  try {
    const initReply = await ctx.reply(marked.parseInline(`... \n`), {
      reply_to_message_id: ctx.message.message_id,
      parse_mode: "HTML",
    });

    // Save the vector store to a directory
    const directory = "data";

    // Load the vector store from the same directory
    const loadedVectorStore = await HNSWLib.load(
      directory,
      new OpenAIEmbeddings()
    );

    const searchResults = await loadedVectorStore.similaritySearch(question, 1);

    const memory = searchResults[0].pageContent;

    console.log("memory: ", memory);

    const updateInterval = setInterval(async () => {
      // console.log("currentMessage: ", currentMessage);
      // console.log("replyedMessage: ", replyedMessage);
      if (currentMessage !== replyedMessage) {
        try {
          const editMessage = await ctx.telegram.editMessageText(
            ctx.chat.id,
            initReply.message_id,
            "0",
            currentMessage
          );
          if (editMessage) {
            replyedMessage = currentMessage;
          }
        } catch (error) {
          console.log(error);
        }
      }
    }, 1000);

    setTimeout(() => {
      clearInterval(updateInterval);
    }, 1000 * 30);

    chain
      .call({
        input: `Your memory of this conversation is: ${memory} \nQuestion: ${question}`,
      })
      .then(async ({ response }) => {
        console.log("response: ", response);

        clearInterval(updateInterval);
        setTimeout(() => {
          try {
            ctx.telegram.editMessageText(
              ctx.chat.id,
              initReply.message_id,
              "0",
              response
            );
          } catch (error) {
            console.log(error);
          }
        }, 1000);

        const docs = [
          new Document({
            metadata: { id: uuidv4() },
            pageContent: `Question: ${question} Answer: ${response}`,
          }),
        ];

        // console.log(docs);

        await loadedVectorStore.addDocuments(docs);

        await loadedVectorStore.save(directory);
      });

    // console.log(res.response);
  } catch (error) {
    console.log(error);
  }
});

bot.launch();

console.log("Bot started", new Date().toLocaleTimeString());

process.once("SIGINT", () => bot.stop("SIGINT"));
process.once("SIGTERM", () => bot.stop("SIGTERM"));
