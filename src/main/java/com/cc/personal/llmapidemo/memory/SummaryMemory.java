package com.cc.personal.llmapidemo.memory;

import com.google.gson.*;
import okhttp3.*;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static com.cc.personal.llmapidemo.constant.Constant.*;

/**
 * 支持摘要压缩的会话记忆管理器
 *
 * 需要注意：1 触发摘要压缩的tokens大小设置；2 压缩时压缩早期的多少轮消息，保留最近的多少轮消息 可以设定一个参数控制
 *          3 注意摘要压缩的时候也是要把摘要和记忆发给chat模型要求压缩，但是其reasoning_tokens可能会很大，超过我们平时设定的chat最大tokens，所以摘要压缩的时候调用chat应该设置更大的token，但是这何尝不是另外的消费呢
 */
public class SummaryMemory {
    private static final String API_URL = LLM_API_URL;
    private static final String API_KEY = LLM_API_KEY;
    private static final String MODEL = LLM_MODEL_NAME;
    //    private static final OkHttpClient client = new OkHttpClient(); 默认的超时是10s容易超时断开
    private static final int MAX_RETRIES = 3;
    private static final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(30,TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .build();
    private static final Gson gson = new Gson();

    /** 触发摘要的 Token 阈值 */
    private final int tokenThreshold;
    /** 保留最近的完整对话轮数 */
    private final int keepRecentRounds;

    /** 会话存储 */
    private final Map<String, List<JsonObject>> store = new HashMap<>();
    /** 摘要存储 */
    private final Map<String, String> summaryStore = new HashMap<>();

    public SummaryMemory(int tokenThreshold, int keepRecentRounds) {
        this.tokenThreshold = tokenThreshold;
        this.keepRecentRounds = keepRecentRounds;
    }

    public void addMessage(String sessionId, String role, String content) {
        store.computeIfAbsent(sessionId, k -> new ArrayList<>())
                .add(message(role, content));

        // 检查是否需要触发摘要压缩
        int totalTokens = estimateTotalTokens(sessionId);
        int messageCount = store.get(sessionId).size();
        int earlyCount = messageCount - keepRecentRounds * 2;
        System.out.println("[Token估算] 当前总tokens: " + totalTokens + " / 阈值: " + tokenThreshold
                + " | 消息数: " + messageCount + " | 可压缩消息数: " + Math.max(0, earlyCount));
        if (totalTokens > tokenThreshold && earlyCount > 0) {
            try {
                compress(sessionId);
            } catch (Exception e) {
                System.err.println("摘要压缩失败：" + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    /**
     * 压缩早期对话为摘要
     */
    private void compress(String sessionId) throws IOException {
        List<JsonObject> allMessages = store.get(sessionId);
        if (allMessages == null || allMessages.size() <= keepRecentRounds * 2) {
            return;
        }

        // 分离：早期消息（要压缩的）+ 最近消息（要保留的）
        int keepCount = keepRecentRounds * 2;
        List<JsonObject> earlyMessages = allMessages.subList(
                0, allMessages.size() - keepCount);
        List<JsonObject> recentMessages = new ArrayList<>(
                allMessages.subList(allMessages.size() - keepCount, allMessages.size()));

        // 构建要压缩的对话文本
        StringBuilder conversationText = new StringBuilder();
        for (JsonObject msg : earlyMessages) {
            String role = msg.get("role").getAsString();
            String content = msg.get("content").getAsString();
            conversationText.append(role).append("：").append(content).append("\n");
        }

        // 获取已有的摘要
        String existingSummary = summaryStore.getOrDefault(sessionId, "");

        // 调用大模型生成摘要
        String summaryPrompt = "请将以下对话历史压缩为一段简洁的摘要，要求：\n" +
                "1. 保留用户的核心意图和关注点\n" +
                "2. 保留所有关键实体（产品名、订单号、日期、金额等）\n" +
                "3. 保留已经确认的结论和决定\n" +
                "4. 保留尚未解决的问题\n" +
                "5. 省略寒暄、重复确认、无关细节\n" +
                "6. 摘要以第三人称描述，控制在 200 字以内\n";

        if (!existingSummary.isEmpty()) {
            summaryPrompt += "\n已有的历史摘要：\n" + existingSummary + "\n";
        }
        summaryPrompt += "\n需要压缩的新对话：\n" + conversationText;

        System.out.println("[摘要压缩] 开始调用大模型生成摘要，早期消息数: " + earlyMessages.size()
                + " | prompt长度: " + summaryPrompt.length() + " 字符");

        // 带重试的摘要生成
        String summary = null;
        for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            try {
                summary = chat(List.of(
                        message("system", "你是一个对话摘要助手，负责将对话历史压缩为简洁的摘要。"),
                        message("user", summaryPrompt)
                ),10000);
            } catch (Exception e) {
                System.err.println("[摘要压缩] 第 " + attempt + " 次调用失败: " + e.getMessage());
            }
            if (summary != null && !summary.trim().isEmpty()) {
                break;
            }
            System.err.println("[摘要压缩] 第 " + attempt + " 次返回为空" +
                    (attempt < MAX_RETRIES ? "，重试中..." : "，已放弃"));
        }

        if (summary == null || summary.trim().isEmpty()) {
            System.err.println("[摘要压缩] 警告：重试 " + MAX_RETRIES + " 次后摘要仍为空，保留原始消息不压缩");
            return;
        }

        // 更新摘要和消息列表
        summaryStore.put(sessionId, summary);
        store.put(sessionId, recentMessages);

        System.out.println("[摘要压缩] 将 " + earlyMessages.size() +
                " 条早期消息压缩为摘要");
        System.out.println("[摘要内容] " + summary);
    }

    /**
     * 构建发送给 API 的完整 messages 数组
     */
    public List<JsonObject> buildMessages(String sessionId,
                                          String systemPrompt,
                                          String currentQuestion) {
        List<JsonObject> messages = new ArrayList<>();
        messages.add(message("system", systemPrompt));

        // 添加摘要（如果有）
        String summary = summaryStore.get(sessionId);
        if (summary != null && !summary.isEmpty()) {
            messages.add(message("system",
                    "【对话背景摘要】" + summary));
        }

        // 添加最近的完整对话
        List<JsonObject> recentMessages = store.getOrDefault(
                sessionId, List.of());
        messages.addAll(recentMessages);

        // 添加当前问题
        messages.add(message("user", currentQuestion));
        return messages;
    }

    private int estimateTotalTokens(String sessionId) {
        List<JsonObject> messages = store.getOrDefault(sessionId, List.of());
        int total = 0;
        for (JsonObject msg : messages) {
            total += estimateTokens(msg.get("content").getAsString());
        }
        return total;
    }

    /** 简单的 Token 估算 */
    static int estimateTokens(String text) {
        if (text == null || text.isEmpty()) return 0;
        int chineseChars = 0, otherChars = 0;
        for (char c : text.toCharArray()) {
            if (Character.UnicodeScript.of(c) == Character.UnicodeScript.HAN) {
                chineseChars++;
            } else if (!Character.isWhitespace(c)) {
                otherChars++;
            }
        }
        return (int) (chineseChars * 1.5 + otherChars / 4.0);
    }

    // chat() 和 message() 方法与前面的 MemoryDemo 相同，此处省略
    private JsonObject message(String role, String content) {
        JsonObject msg = new JsonObject();
        msg.addProperty("role", role);
        msg.addProperty("content", content);
        return msg;
    }

    static String chat(List<JsonObject> messages,int max_tokens) throws IOException {
        JsonObject body = new JsonObject();
        body.addProperty("model", MODEL);
        body.addProperty("temperature", 0.1);
        body.addProperty("max_tokens", max_tokens !=0 ? max_tokens: 512); // 这里需要注意一下，摘要推理的过程花了511个token导致超出返回为空
        JsonArray messagesArray = new JsonArray();
        for (JsonObject msg : messages) {
            messagesArray.add(msg);
        }
        body.add("messages", messagesArray);

        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(body.toString(),
                        MediaType.parse("application/json")))
                .build();

        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body().string();
            if (!response.isSuccessful()) {
                throw new IOException("API返回非200状态码: " + response.code() + " | 响应: " + responseBody);
            }
            JsonObject json = gson.fromJson(responseBody, JsonObject.class);

            // 详细日志：打印API完整响应
            System.out.println("[API响应] " + responseBody);

            JsonArray choices = json.getAsJsonArray("choices");
            if (choices == null || choices.size() == 0) {
                throw new IOException("API返回空choices: " + responseBody);
            }
            JsonObject choice = choices.get(0).getAsJsonObject();
            String content = choice.getAsJsonObject("message").get("content").getAsString();
            String finishReason = choice.has("finish_reason") ?
                    choice.get("finish_reason").getAsString() : "unknown";
            System.out.println("[API] finish_reason=" + finishReason + " | content长度=" +
                    (content != null ? content.length() : 0));
            return content;
        }
    }


}
