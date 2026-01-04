import OpenAI from 'openai';
import BaseLLM from '../../base/llm';
import { zodTextFormat, zodResponseFormat } from 'openai/helpers/zod';
import {
  GenerateObjectInput,
  GenerateOptions,
  GenerateTextInput,
  GenerateTextOutput,
  StreamTextOutput,
  ToolCall,
} from '../../types';
import { parse } from 'partial-json';
import z from 'zod';
import {
  ChatCompletionAssistantMessageParam,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionToolMessageParam,
} from 'openai/resources/index.mjs';
import { Message } from '@/lib/types';
import { repairJson } from '@toolsycc/json-repair';

type OpenAIConfig = {
  apiKey: string;
  model: string;
  baseURL?: string;
  options?: GenerateOptions;
};

class OpenAILLM extends BaseLLM<OpenAIConfig> {
  openAIClient: OpenAI;

  constructor(protected config: OpenAIConfig) {
    super(config);

    this.openAIClient = new OpenAI({
      apiKey: this.config.apiKey,
      baseURL: this.config.baseURL || 'https://api.openai.com/v1',
    });
  }

  convertToOpenAIMessages(messages: Message[]): ChatCompletionMessageParam[] {
    return messages.map((msg) => {
      if (msg.role === 'tool') {
        return {
          role: 'tool',
          tool_call_id: msg.id,
          content: msg.content,
        } as ChatCompletionToolMessageParam;
      } else if (msg.role === 'assistant') {
        return {
          role: 'assistant',
          content: msg.content,
          ...(msg.tool_calls &&
            msg.tool_calls.length > 0 && {
              tool_calls: msg.tool_calls?.map((tc) => ({
                id: tc.id,
                type: 'function',
                function: {
                  name: tc.name,
                  arguments: JSON.stringify(tc.arguments),
                },
              })),
            }),
        } as ChatCompletionAssistantMessageParam;
      }

      return msg;
    });
  }

  async generateText(input: GenerateTextInput): Promise<GenerateTextOutput> {
    // Check if this is an OpenRouter model
    const isOpenRouter =
      this.config.model?.startsWith('openrouter/') ||
      this.config.baseURL?.includes('openrouter');

    // DeepSeek models support tool calling via OpenRouter (confirmed in their docs)
    const isDeepSeek =
      this.config.model?.toLowerCase().includes('deepseek') ||
      this.config.model?.toLowerCase().includes('nex-agi');

    // Only disable tools for non-DeepSeek OpenRouter models
    const disableTools = isOpenRouter && !isDeepSeek;

    const openaiTools: ChatCompletionTool[] = [];

    if (!disableTools) {
      input.tools?.forEach((tool) => {
        openaiTools.push({
          type: 'function',
          function: {
            name: tool.name,
            description: tool.description,
            parameters: z.toJSONSchema(tool.schema),
          },
        });
      });
    }

    const response = await this.openAIClient.chat.completions.create({
      model: this.config.model,
      tools: openaiTools.length > 0 ? openaiTools : undefined,
      messages: this.convertToOpenAIMessages(input.messages),
      temperature:
        input.options?.temperature ?? this.config.options?.temperature ?? 1.0,
      top_p: input.options?.topP ?? this.config.options?.topP,
      max_completion_tokens:
        input.options?.maxTokens ?? this.config.options?.maxTokens,
      stop: input.options?.stopSequences ?? this.config.options?.stopSequences,
      frequency_penalty:
        input.options?.frequencyPenalty ??
        this.config.options?.frequencyPenalty,
      presence_penalty:
        input.options?.presencePenalty ?? this.config.options?.presencePenalty,
    });

    if (response.choices && response.choices.length > 0) {
      return {
        content: response.choices[0].message.content!,
        toolCalls:
          response.choices[0].message.tool_calls
            ?.map((tc) => {
              if (tc.type === 'function') {
                return {
                  name: tc.function.name,
                  id: tc.id,
                  arguments: JSON.parse(tc.function.arguments),
                };
              }
            })
            .filter((tc) => tc !== undefined) || [],
        additionalInfo: {
          finishReason: response.choices[0].finish_reason,
        },
      };
    }

    throw new Error('No response from OpenAI');
  }

  async *streamText(
    input: GenerateTextInput,
  ): AsyncGenerator<StreamTextOutput> {
    // Check if this is an OpenRouter model
    const isOpenRouter =
      this.config.model?.startsWith('openrouter/') ||
      this.config.baseURL?.includes('openrouter');

    // DeepSeek models support tool calling via OpenRouter (confirmed in their docs)
    const isDeepSeek =
      this.config.model?.toLowerCase().includes('deepseek') ||
      this.config.model?.toLowerCase().includes('nex-agi');

    // Only disable tools for non-DeepSeek OpenRouter models
    const disableTools = isOpenRouter && !isDeepSeek;

    const openaiTools: ChatCompletionTool[] = [];

    if (!disableTools) {
      input.tools?.forEach((tool) => {
        openaiTools.push({
          type: 'function',
          function: {
            name: tool.name,
            description: tool.description,
            parameters: z.toJSONSchema(tool.schema),
          },
        });
      });
    }

    const stream = await this.openAIClient.chat.completions.create({
      model: this.config.model,
      messages: this.convertToOpenAIMessages(input.messages),
      tools: openaiTools.length > 0 ? openaiTools : undefined,
      temperature:
        input.options?.temperature ?? this.config.options?.temperature ?? 1.0,
      top_p: input.options?.topP ?? this.config.options?.topP,
      max_completion_tokens:
        input.options?.maxTokens ?? this.config.options?.maxTokens,
      stop: input.options?.stopSequences ?? this.config.options?.stopSequences,
      frequency_penalty:
        input.options?.frequencyPenalty ??
        this.config.options?.frequencyPenalty,
      presence_penalty:
        input.options?.presencePenalty ?? this.config.options?.presencePenalty,
      stream: true,
    });

    let recievedToolCalls: { name: string; id: string; arguments: string }[] =
      [];

    for await (const chunk of stream) {
      if (chunk.choices && chunk.choices.length > 0) {
        const toolCalls = chunk.choices[0].delta.tool_calls;
        let parsedToolCalls: ToolCall[] = [];

        if (toolCalls) {
          for (const tc of toolCalls) {
            try {
              if (!recievedToolCalls[tc.index]) {
                const call = {
                  name: tc.function?.name || '',
                  id: tc.id || `tool_${tc.index}`,
                  arguments: tc.function?.arguments || '',
                };
                recievedToolCalls.push(call);
                parsedToolCalls.push({ ...call, arguments: parse(call.arguments || '{}') as Record<string, unknown> });
              } else {
                const existingCall = recievedToolCalls[tc.index];
                existingCall.arguments += tc.function?.arguments || '';
                parsedToolCalls.push({
                  ...existingCall,
                  arguments: parse(existingCall.arguments || '{}') as Record<string, unknown>,
                });
              }
            } catch (parseErr) {
              console.warn('Failed to parse tool call arguments:', parseErr);
              // Return empty object for malformed JSON
              const call = recievedToolCalls[tc.index] || { name: '', id: `tool_${tc.index}`, arguments: '' };
              parsedToolCalls.push({ name: call.name, id: call.id, arguments: {} });
            }
          }
        }

        yield {
          contentChunk: chunk.choices[0].delta.content || '',
          toolCallChunk: parsedToolCalls,
          done: chunk.choices[0].finish_reason !== null,
          additionalInfo: {
            finishReason: chunk.choices[0].finish_reason,
          },
        };
      }
    }
  }

  private stripMarkdownCodeFence(text: string): string {
    // Strip ```json ... ``` or ``` ... ``` wrappers
    const match = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
    return match ? match[1] : text;
  }

  async generateObject<T>(input: GenerateObjectInput): Promise<T> {
    // Use json_object mode for Groq/OpenRouter (don't support json_schema on most models)
    const needsJsonObjectMode =
      this.config.baseURL?.includes('groq') ||
      this.config.model?.startsWith('groq/') ||
      this.config.model?.startsWith('openrouter/');

    // For json_object mode, inject schema into system prompt since it won't be enforced
    let messages = this.convertToOpenAIMessages(input.messages);
    if (needsJsonObjectMode) {
      const schemaJson = JSON.stringify(z.toJSONSchema(input.schema), null, 2);
      const schemaPrompt = `You MUST respond with valid JSON matching this exact schema. All fields are required unless marked optional:\n${schemaJson}\n\nRespond ONLY with the JSON object, no additional text.`;

      // Prepend schema instruction to first message or add as system message
      if (messages.length > 0 && messages[0].role === 'system') {
        messages[0] = {
          ...messages[0],
          content: `${messages[0].content}\n\n${schemaPrompt}`,
        };
      } else {
        messages = [{ role: 'system', content: schemaPrompt }, ...messages];
      }
    }

    const response = await this.openAIClient.chat.completions.create({
      messages,
      model: this.config.model,
      temperature:
        input.options?.temperature ?? this.config.options?.temperature ?? 1.0,
      top_p: input.options?.topP ?? this.config.options?.topP,
      max_completion_tokens:
        input.options?.maxTokens ?? this.config.options?.maxTokens,
      stop: input.options?.stopSequences ?? this.config.options?.stopSequences,
      frequency_penalty:
        input.options?.frequencyPenalty ??
        this.config.options?.frequencyPenalty,
      presence_penalty:
        input.options?.presencePenalty ?? this.config.options?.presencePenalty,
      response_format: needsJsonObjectMode ? { type: 'json_object' } : zodResponseFormat(input.schema, 'object'),
    });

    if (response.choices && response.choices.length > 0) {
      try {
        const rawContent = response.choices[0].message.content!;
        const strippedContent = this.stripMarkdownCodeFence(rawContent.trim());
        const parsed = JSON.parse(
          repairJson(strippedContent, {
            extractJson: true,
          }) as string,
        );

        // For json_object mode, use safeParse and provide defaults for missing fields
        if (needsJsonObjectMode) {
          const result = input.schema.safeParse(parsed);
          if (result.success) {
            return result.data as T;
          }
          // Log validation errors but try to return partial data with defaults
          console.warn('Schema validation failed, attempting partial parse:', result.error.issues);
          // Return parsed object as-is, let caller handle missing fields
          return parsed as T;
        }

        return input.schema.parse(parsed) as T;
      } catch (err) {
        throw new Error(`Error parsing response from OpenAI: ${err}`);
      }
    }

    throw new Error('No response from OpenAI');
  }

  async *streamObject<T>(input: GenerateObjectInput): AsyncGenerator<T> {
    let recievedObj: string = '';

    const stream = this.openAIClient.responses.stream({
      model: this.config.model,
      input: input.messages,
      temperature:
        input.options?.temperature ?? this.config.options?.temperature ?? 1.0,
      top_p: input.options?.topP ?? this.config.options?.topP,
      max_completion_tokens:
        input.options?.maxTokens ?? this.config.options?.maxTokens,
      stop: input.options?.stopSequences ?? this.config.options?.stopSequences,
      frequency_penalty:
        input.options?.frequencyPenalty ??
        this.config.options?.frequencyPenalty,
      presence_penalty:
        input.options?.presencePenalty ?? this.config.options?.presencePenalty,
      text: {
        format: zodTextFormat(input.schema, 'object'),
      },
    });

    for await (const chunk of stream) {
      if (chunk.type === 'response.output_text.delta' && chunk.delta) {
        recievedObj += chunk.delta;

        try {
          yield parse(recievedObj) as T;
        } catch (err) {
          console.log('Error parsing partial object from OpenAI:', err);
          yield {} as T;
        }
      } else if (chunk.type === 'response.output_text.done' && chunk.text) {
        try {
          yield parse(chunk.text) as T;
        } catch (err) {
          throw new Error(`Error parsing response from OpenAI: ${err}`);
        }
      }
    }
  }
}

export default OpenAILLM;
