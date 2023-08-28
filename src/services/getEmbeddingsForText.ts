import { TextEmbedding } from "../types/file";
import { chunkText } from "./chunkText";
import { embedding } from "./replicate";

// There isn't a good JS tokenizer at the moment, so we are using this approximation of 4 characters per token instead. This might break for some languages.
const MAX_CHAR_LENGTH = 500 * 4;

// This function takes a text and returns an array of embeddings for each chunk of the text
// The text is split into chunks of a given maximum character length
// The embeddings are computed in batches of a given size
export async function getEmbeddingsForText({
  text,
  maxCharLength = MAX_CHAR_LENGTH,
  batchSize = 10,
}: {
  text: string;
  maxCharLength?: number;
  batchSize?: number;
}): Promise<TextEmbedding[]> {
  const textChunks = chunkText({ text, maxCharLength });

  const batches = [];
  for (let i = 0; i < textChunks.length; i += batchSize) {
    batches.push(textChunks.slice(i, i + batchSize));
  }

  console.log(batches.length);

  try {
    const batchPromises = batches.map((batch) => embedding({ input: batch }));

    const embeddings = (await Promise.all(batchPromises)).flat();

    const textEmbeddings = embeddings.map((embedding, index) => ({
      embedding,
      text: textChunks[index],
    }));

    console.log(textEmbeddings[textEmbeddings.length - 1])

    return textEmbeddings;
  } catch (error: any) {
    console.log("Error: ", error);
    return [];
  }
}
