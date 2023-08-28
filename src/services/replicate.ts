import Replicate from "replicate";
import { ReplicateStream } from "ai";

type EmbeddingOptions = {
  input: string | string[];
  model?: `${string}/${string}:${string}`;
};

type EmbeddingResult = {
  embedding: number[];
};

// This file contains utility functions for interacting with the Replicate API

if (!process.env.REPLICATE_API_TOKEN) {
  throw new Error("Missing REPLICATE_API_TOKEN environment variable");
}

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

export async function completionStream(inputPrompt: string) {
  try {
    const prediction = await replicate.predictions.create({
      // https://replicate.com/a16z-infra/llama-2-13b-chat/versions
      version: "9dff94b1bed5af738655d4a7cbcdcde2bd503aa85c94334fe1f42af7f3dd5ee3",
      input: { prompt: inputPrompt, max_new_tokens: 1000 },
      stream: true,
    });
    if (!prediction.urls.stream) {
      throw new Error("The stream's URL is null or undefined");
    }
    return await ReplicateStream(prediction);
  } catch (error) {
    throw error;
  }
}

export async function embedding({
  input,
  model = "replicate/all-mpnet-base-v2:b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305",
}: EmbeddingOptions): Promise<number[][]> {
  const jsonInput = JSON.stringify(input);
  const result = await replicate.run(
    model,
    {
      input: {
        text_batch: jsonInput
      }
    }
  ) as EmbeddingResult[];
  return result.map(item => item.embedding);
}