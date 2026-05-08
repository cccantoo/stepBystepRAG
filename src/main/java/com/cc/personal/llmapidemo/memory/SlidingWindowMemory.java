package com.cc.personal.llmapidemo.memory;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import okhttp3.MediaType;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;
import java.util.*;

import static com.cc.personal.llmapidemo.constant.Constant.LLM_MODEL_NAME;

/**
 * 滑动窗口会话记忆管理器
 */
public class SlidingWindowMemory {

    /** 最大保留轮数（1 轮 = 1 个 user + 1 个 assistant） */
    private final int maxRounds;

    /** 会话存储：sessionId → 消息列表 */
    private final Map<String, List<JsonObject>> store = new HashMap<>();

    public SlidingWindowMemory(int maxRounds) {
        this.maxRounds = maxRounds;
    }

    /**
     * 添加一条消息
     */
    public void addMessage(String sessionId, String role, String content) {
        store.computeIfAbsent(sessionId, k -> new ArrayList<>())
                .add(message(role, content));
    }

    /**
     * 获取最近 N 轮的消息（滑动窗口）
     * 一轮 = user + assistant 两条消息
     */
    public List<JsonObject> getRecentMessages(String sessionId) {
        List<JsonObject> allMessages = store.getOrDefault(sessionId, List.of());
        if (allMessages.isEmpty()) {
            return List.of();
        }

        // 计算要保留的消息数量：maxRounds 轮 × 2 条/轮
        int keepCount = maxRounds * 2;
        if (allMessages.size() <= keepCount) {
            return new ArrayList<>(allMessages);
        }

        // 只保留最近的 keepCount 条消息
        return new ArrayList<>(
                allMessages.subList(allMessages.size() - keepCount, allMessages.size())
        );
    }

    /**
     * 构建发送给 API 的完整 messages 数组
     */
    public List<JsonObject> buildMessages(String sessionId,
                                          String systemPrompt,
                                          String currentQuestion) {
        List<JsonObject> messages = new ArrayList<>();
        messages.add(message("system", systemPrompt));
        messages.addAll(getRecentMessages(sessionId));
        messages.add(message("user", currentQuestion));
        return messages;
    }

    private JsonObject message(String role, String content) {
        JsonObject msg = new JsonObject();
        msg.addProperty("role", role);
        msg.addProperty("content", content);
        return msg;
    }



}
