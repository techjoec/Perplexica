import z from 'zod';
import { ResearchAction } from '../../types';
import { getCloudflareRAGURL } from '@/lib/config/serverRegistry';
import { Chunk, SearchResultsResearchBlock } from '@/lib/types';

const actionSchema = z.object({
  type: z.literal('caption_search'),
  query: z
    .string()
    .describe('Search query for CraftyPanda historical captions and craft content'),
});

interface CloudflareSearchResult {
  id: string;
  content: string;
  folderName: string;
  platform?: string;
  holiday?: string;
  imageCount: number;
  score: number;
}

interface CloudflareSearchResponse {
  query: string;
  results: CloudflareSearchResult[];
  total: number;
}

const captionSearchPrompt = `
Use this tool to search CraftyPanda's historical Instagram captions when the user asks about:
- Past products, projects, or crafts the business has made
- Seasonal or holiday craft themes (Christmas, Halloween, etc.)
- Product descriptions, pricing patterns, or promotional content
- Craft-related information specific to CraftyPanda

This searches a curated database of historical posts with semantic similarity.
Always use this alongside web search when the query relates to crafts, decorations, or CraftyPanda-specific content.
`;

const captionSearchAction: ResearchAction<typeof actionSchema> = {
  name: 'caption_search',
  schema: actionSchema,
  getToolDescription: () =>
    "Search CraftyPanda's historical Instagram captions for craft-related content, product descriptions, and seasonal themes.",
  getDescription: () => captionSearchPrompt,

  // Auto-include: run when 'web' is enabled (no separate 'captions' toggle needed)
  enabled: (config) =>
    config.sources.includes('web') &&
    !config.classification.classification.skipSearch,

  execute: async (input, additionalConfig) => {
    const ragURL = getCloudflareRAGURL();

    if (!ragURL) {
      console.warn('captionSearch: CLOUDFLARE_RAG_URL not configured');
      return { type: 'search_results', results: [] };
    }

    const researchBlock = additionalConfig.session.getBlock(
      additionalConfig.researchBlockId,
    );

    // Show searching status in UI
    if (researchBlock?.type === 'research') {
      researchBlock.data.subSteps.push({
        id: crypto.randomUUID(),
        type: 'searching',
        searching: [`CraftyPanda: ${input.query}`],
      });

      additionalConfig.session.updateBlock(additionalConfig.researchBlockId, [
        {
          op: 'replace',
          path: '/data/subSteps',
          value: researchBlock.data.subSteps,
        },
      ]);
    }

    try {
      const res = await fetch(`${ragURL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input.query, limit: 5, threshold: 0.5 }),
      });

      if (!res.ok) {
        console.error('captionSearch: API error', res.status);
        return { type: 'search_results', results: [] };
      }

      const data: CloudflareSearchResponse = await res.json();

      const results: Chunk[] = data.results.map((r) => ({
        content: r.content,
        metadata: {
          title: `CraftyPanda: ${r.folderName.replace(/_/g, ' ')}`,
          url: `craftypanda://caption/${r.id}`,
          source: 'CraftyPanda Historical',
          platform: r.platform,
          holiday: r.holiday,
          imageCount: r.imageCount,
          score: r.score,
        },
      }));

      // Emit search results to research block UI
      const searchResultsBlockId = crypto.randomUUID();

      if (researchBlock?.type === 'research') {
        researchBlock.data.subSteps.push({
          id: searchResultsBlockId,
          type: 'search_results',
          reading: results,
        } as SearchResultsResearchBlock);

        additionalConfig.session.updateBlock(additionalConfig.researchBlockId, [
          {
            op: 'replace',
            path: '/data/subSteps',
            value: researchBlock.data.subSteps,
          },
        ]);
      }

      return { type: 'search_results', results };
    } catch (err) {
      console.error('captionSearch: fetch error', err);
      return { type: 'search_results', results: [] };
    }
  },
};

export default captionSearchAction;
