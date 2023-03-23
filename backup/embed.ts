import dotenv from "dotenv";
import { Document } from "langchain/document";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { HNSWLib } from "langchain/vectorstores";
import { v4 as uuidv4 } from "uuid";

dotenv.config();

export const run = async () => {
  // Create a vector store through any method, here from texts as an example
  //   const vectorStore = await HNSWLib.fromTexts(
  //     ["Hello world", "Bye bye", "hello nice world"],
  //     [{ id: 2 }, { id: 1 }, { id: 3 }],
  //     new OpenAIEmbeddings()
  //   );

  // Save the vector store to a directory
  const directory = "data";
  //   await vectorStore.save(directory);

  // Load the vector store from the same directory
  const loadedVectorStore = await HNSWLib.load(
    directory,
    new OpenAIEmbeddings()
  );

  //   const docs = [
  //     new Document({
  //       metadata: { id: uuidv4() },
  //       pageContent: "pinecone is a vector db",
  //     }),
  //   ];

  //   await loadedVectorStore.addDocuments(docs);

  //   await loadedVectorStore.save(directory);

  // vectorStore and loadedVectorStore are identical

  const result = await loadedVectorStore.similaritySearch("pinecone", 1);
  console.log(result);
};

await run();
