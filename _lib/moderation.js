const OPENAI_URL = "https://api.openai.com/v1/moderations";
const HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions";
const HF_NSFW_URL = "https://router.huggingface.co/hf-inference/models/Falconsai/nsfw_image_detection";

const bool = (v) => v === true || String(v).toLowerCase() === "true";

function imageBase64(dataUrl) {
  return String(dataUrl || "").replace(/^data:image\/[a-z0-9.+-]+;base64,/i, "");
}

async function postJson(url, body, headers = {}, timeoutMs = 25000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await response.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
    if (!response.ok) {
      const message = data?.error?.message || data?.error || data?.message || response.statusText;
      throw new Error(`${response.status}: ${message}`);
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

async function openAiModeration(text, photo) {
  if (!process.env.OPENAI_API_KEY) throw new Error("OPENAI_API_KEY is not configured.");
  const content = [{ type: "text", text }];
  if (photo) content.push({ type: "image_url", image_url: { url: photo } });

  const data = await postJson(
    OPENAI_URL,
    {
      model: process.env.OPENAI_MODERATION_MODEL || "omni-moderation-latest",
      input: [{ role: "user", content }],
    },
    { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` }
  );

  const result = data.results?.[0] || {};
  return {
    provider: "openai",
    model: process.env.OPENAI_MODERATION_MODEL || "omni-moderation-latest",
    enabled: true,
    flagged: Boolean(result.flagged),
    categories: result.categories || {},
    categoryScores: result.category_scores || {},
  };
}

async function qwenCivicCheck(text, category, photo) {
  if (!process.env.HF_TOKEN) throw new Error("HF_TOKEN is not configured.");
  const content = [
    {
      type: "text",
      text:
        "You are the Civic Pulse verification model. Analyze whether this report is a genuine civic/local public issue. " +
        "Check whether the image, description and selected category are consistent. Do not judge the person. " +
        "Return ONLY valid JSON with keys: is_civic_issue (boolean), image_matches_text (boolean), " +
        "category_matches (boolean), confidence (number 0-1), reason (string), suggested_category (string).\n\n" +
        `Description: ${text}\nSelected category: ${category}`,
    },
  ];
  if (photo) content.push({ type: "image_url", image_url: { url: photo } });

  const data = await postJson(
    HF_CHAT_URL,
    {
      model: process.env.HF_QWEN_MODEL || "Qwen/Qwen2.5-VL-3B-Instruct",
      messages: [
        {
          role: "user",
          content,
        },
      ],
      temperature: 0,
      max_tokens: 400,
    },
    { Authorization: `Bearer ${process.env.HF_TOKEN}` },
    40000
  );

  const raw = data.choices?.[0]?.message?.content || "";
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("Qwen returned an invalid moderation response.");
  const parsed = JSON.parse(match[0]);

  return {
    provider: "huggingface",
    model: process.env.HF_QWEN_MODEL || "Qwen/Qwen2.5-VL-3B-Instruct",
    enabled: true,
    ...parsed,
    is_civic_issue: bool(parsed.is_civic_issue),
    image_matches_text: bool(parsed.image_matches_text),
    category_matches: bool(parsed.category_matches),
    confidence: Number(parsed.confidence || 0),
  };
}

async function hfNsfwCheck(photo) {
  if (!photo) {
    return { provider: "huggingface", model: "Falconsai/nsfw_image_detection", enabled: false, skipped: true };
  }
  if (!process.env.HF_TOKEN) throw new Error("HF_TOKEN is not configured.");

  const data = await postJson(
    process.env.HF_NSFW_URL || HF_NSFW_URL,
    {
      inputs: imageBase64(photo),
      parameters: { function_to_apply: "softmax", top_k: 2 },
    },
    { Authorization: `Bearer ${process.env.HF_TOKEN}` },
    30000
  );

  const predictions = Array.isArray(data) ? data : Array.isArray(data?.[0]) ? data[0] : [];
  const nsfw = predictions.find((x) => String(x.label).toLowerCase() === "nsfw");
  const normal = predictions.find((x) => String(x.label).toLowerCase() === "normal");

  return {
    provider: "huggingface",
    model: "Falconsai/nsfw_image_detection",
    enabled: true,
    flagged: Number(nsfw?.score || 0) >= Number(process.env.NSFW_THRESHOLD || 0.5),
    nsfw_confidence: Number(nsfw?.score || 0),
    normal_confidence: Number(normal?.score || 0),
    predictions,
  };
}

async function aiGeneratedCheck(photo) {
  if (!photo) {
    return { provider: "external", enabled: false, skipped: true };
  }
  const url = process.env.AI_IMAGE_DETECTOR_URL;
  if (!url) throw new Error("AI_IMAGE_DETECTOR_URL is not configured.");

  const data = await postJson(
    url,
    {
      image: photo,
      image_base64: imageBase64(photo),
    },
    process.env.AI_IMAGE_DETECTOR_TOKEN
      ? { Authorization: `Bearer ${process.env.AI_IMAGE_DETECTOR_TOKEN}` }
      : {},
    40000
  );

  const confidence = Number(
    data.ai_generated_confidence ??
    data.ai_generated_probability ??
    data.probability ??
    data.score ??
    0
  );

  return {
    provider: data.provider || "external",
    model: data.model || process.env.AI_IMAGE_DETECTOR_MODEL || "Smaug-AI-Image-Detector",
    enabled: true,
    ai_generated_confidence: confidence,
    real_image_confidence: Number(data.real_image_confidence ?? (1 - confidence) * 100),
    ai_detection_status: data.ai_detection_status || (confidence >= 0.5 ? "likely_ai_generated" : "likely_real"),
    raw: data,
  };
}

export async function moderateIssue({ text, category, photo }) {
  const requireAll = process.env.MODERATION_REQUIRE_ALL === "true";
  const jobs = await Promise.allSettled([
    openAiModeration(text, photo),
    qwenCivicCheck(text, category, photo),
    hfNsfwCheck(photo),
    aiGeneratedCheck(photo),
  ]);

  const names = ["safety", "civic_issue", "nsfw", "ai_generated"];
  const result = { safety: null, civic_issue: null, nsfw: null, ai_generated: null, errors: [] };

  jobs.forEach((job, index) => {
    if (job.status === "fulfilled") result[names[index]] = job.value;
    else result.errors.push({ model: names[index], message: job.reason?.message || "Model request failed." });
  });

  const safetyBlocked = Boolean(result.safety?.flagged);
  const civicBlocked = result.civic_issue
    ? !(result.civic_issue.is_civic_issue && result.civic_issue.image_matches_text && result.civic_issue.category_matches)
    : requireAll;
  const nsfwBlocked = Boolean(result.nsfw?.flagged);
  const aiBlocked = result.ai_generated
    ? Number(result.ai_generated.ai_generated_confidence || 0) >= Number(process.env.AI_GENERATED_THRESHOLD || 0.9)
    : requireAll && Boolean(photo);

  const canPost = !safetyBlocked && !civicBlocked && !nsfwBlocked && !aiBlocked && (!requireAll || result.errors.length === 0);

  const problems = [];
  if (safetyBlocked) problems.push("The content was flagged by the safety moderation service.");
  if (civicBlocked) problems.push(result.civic_issue?.reason || "The image, description, or category does not appear to describe a matching civic issue.");
  if (nsfwBlocked) problems.push("The uploaded image appears to contain NSFW content.");
  if (aiBlocked) problems.push("The uploaded image has a high probability of being AI-generated.");
  if (requireAll && result.errors.length) problems.push("One or more required moderation services could not be reached.");

  return {
    verification_status: canPost ? "VERIFIED" : "NEEDS_CORRECTION",
    message: canPost
      ? "The image, description and category passed the moderation checks. The report is ready for posting."
      : "Please correct the highlighted issue before posting.",
    problems,
    can_post: canPost,
    checks: result,
  };
}

export async function moderateComment(text) {
  if (!process.env.OPENAI_API_KEY) return { can_post: true, skipped: true };
  const result = await openAiModeration(text, null);
  return {
    can_post: !result.flagged,
    verification_status: result.flagged ? "NEEDS_CORRECTION" : "VERIFIED",
    problems: result.flagged ? ["The comment was flagged by the safety moderation service."] : [],
    checks: { safety: result },
  };
}
