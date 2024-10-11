#!/bin/bash

# Create a directory for the downloads
mkdir -p research_papers
cd research_papers

# Set the timeout in seconds
TIMEOUT=30

# Function to download a paper with timeout
download_paper() {
    title="$1"
    url="$2"
    filename=$(echo "$title" | sed 's/[^a-zA-Z0-9]/_/g').pdf
    echo "Attempting to download: $title"
    wget --timeout=$TIMEOUT --tries=1 -O "$filename" "$url"
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded: $title"
    else
        echo "Failed to download within $TIMEOUT seconds: $title"
        rm -f "$filename"  # Remove the empty or partially downloaded file
    fi
}

# List of papers to download
papers=(
    "The RefinedWeb Dataset for Falcon LLM|https://arxiv.org/pdf/2306.01116.pdf"
    "Dolma: an Open Corpus|https://arxiv.org/pdf/2402.00159.pdf"
    "Finetuned Language Models Are Zero-Shot Learners|https://arxiv.org/pdf/2109.01652.pdf"
    "RedPajama-Data-v2|https://arxiv.org/pdf/2402.01064.pdf"
    "Measuring Massive Multitask Language Understanding|https://arxiv.org/pdf/2009.03300.pdf"
    "MMLU-Pro|https://arxiv.org/pdf/2402.14058.pdf"
    "Judging LLM-as-a-Judge|https://arxiv.org/pdf/2306.05685.pdf"
    "Language Models are Few-Shot Learners|https://arxiv.org/pdf/2005.14165.pdf"
    "In-context Learning and Induction Heads|https://arxiv.org/pdf/2209.11895.pdf"
    "Rethinking the Role of Demonstrations|https://arxiv.org/pdf/2202.12837.pdf"
    "Transformers Learn In-Context by Gradient Descent|https://arxiv.org/pdf/2306.09431.pdf"
    "Few-Shot Parameter-Efficient Fine-Tuning|https://arxiv.org/pdf/2205.05638.pdf"
    "Many-Shot In-Context Learning|https://arxiv.org/pdf/2402.04733.pdf"
    "Chain-of-thought prompting|https://arxiv.org/pdf/2201.11903.pdf"
    "Self-Consistency Improves Chain of Thought Reasoning|https://arxiv.org/pdf/2203.11171.pdf"
    "Towards Revealing the Mystery behind Chain of Thought|https://arxiv.org/pdf/2305.16582.pdf"
    "Scaling Laws for Neural Language Models|https://arxiv.org/pdf/2001.08361.pdf"
    "Data Mixing Laws|https://arxiv.org/pdf/2310.03828.pdf"
    "Training Compute-Optimal Large Language Models|https://arxiv.org/pdf/2203.15556.pdf"
    "Emergent Abilities of Large Language Models|https://arxiv.org/pdf/2206.07682.pdf"
    "Are Emergent Abilities of Large Language Models a Mirage|https://arxiv.org/pdf/2304.15004.pdf"
    "Train Short, Test Long|https://arxiv.org/pdf/2108.12409.pdf"
    "RoFormer|https://arxiv.org/pdf/2104.09864.pdf"
    "The Impact of Positional Encoding on Length Generalization|https://arxiv.org/pdf/2305.19466.pdf"
    "Exploring the Limits of Transfer Learning|https://arxiv.org/pdf/1910.10683.pdf"
    "YaRN: Efficient Context Window Extension|https://arxiv.org/pdf/2309.00071.pdf"
    "Efficient Streaming Language Models|https://arxiv.org/pdf/2309.17453.pdf"
    "LLM Maybe LongLM|https://arxiv.org/pdf/2305.07759.pdf"
    "LM-Infinite|https://arxiv.org/pdf/2308.16137.pdf"
    "Efficient Memory Management for Large Language Model Serving|https://arxiv.org/pdf/2309.06180.pdf"
    "Llama 2: Open Foundation and Fine-Tuned Chat Models|https://arxiv.org/pdf/2307.09288.pdf"
    "DeepSeek-V2|https://arxiv.org/pdf/2402.08598.pdf"
    "Flashattention: Fast and memory-efficient exact attention|https://arxiv.org/pdf/2205.14135.pdf"
    "Flashattention-2|https://arxiv.org/pdf/2307.08691.pdf"
    "Flashattention-3|https://arxiv.org/pdf/2403.09852.pdf"
    "Theory, Analysis, and Best Practices for Sigmoid Self-Attention|https://arxiv.org/pdf/2402.03738.pdf"
    "Jamba-1.5|https://arxiv.org/pdf/2402.19046.pdf"
    "Transformers are SSMs|https://arxiv.org/pdf/2402.18231.pdf"
    "RWKV: Reinventing RNNs for the Transformer Era|https://arxiv.org/pdf/2305.13048.pdf"
    "Griffin|https://arxiv.org/pdf/2402.19427.pdf"
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks|https://arxiv.org/pdf/2005.11401.pdf"
    "Retrieval meets Long Context Large Language Models|https://arxiv.org/pdf/2310.03025.pdf"
    "DeepSeekMath|https://arxiv.org/pdf/2402.03300.pdf"
    "Code Llama: Open Foundation Models for Code|https://arxiv.org/pdf/2308.12950.pdf"
    "BioMistral|https://arxiv.org/pdf/2402.10373.pdf"
    "HyenaDNA|https://arxiv.org/pdf/2306.15794.pdf"
)

# Download papers
for paper in "${papers[@]}"; do
    IFS='|' read -r title url <<< "$paper"
    download_paper "$title" "$url"
    sleep 2  # Add a small delay between downloads to be polite to the server
done

# Download additional resources with timeout
echo "Downloading additional resources..."
wget --timeout=$TIMEOUT --tries=1 -O open_llm_leaderboard.html "https://huggingface.co/spaces/open-llm-leaderboard/blog" || echo "Failed to download open_llm_leaderboard"
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git || echo "Failed to clone lm-evaluation-harness"
wget --timeout=$TIMEOUT --tries=1 -O chatbot_arena_leaderboard.html "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard" || echo "Failed to download chatbot_arena_leaderboard"

echo "Download process completed. Please check the 'research_papers' directory for the downloaded files."
