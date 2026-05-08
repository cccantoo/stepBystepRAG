package com.cc.personal.llmapidemo.sse;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import okhttp3.*;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

import static com.cc.personal.llmapidemo.constant.Constant.*;

/**
 *   ---
 *   SSE 流式客户端（SseStreamClient）代码解析
 *
 *   一、为什么需要 SSE 流式调用？
 *   普通的 LLM 调用是同步阻塞的：发请求 → 等几秒 → 一次性返回完整答案。用户体验上就是一个"转圈等待"。
 *   SSE（Server-Sent Events）流式调用是边生成边返回：每生成一个 token 就立刻推给客户端，用户看到文字一个一个"蹦出来"，就像打字效果。
 *
 *   同步模式：  [发送] ──── 等待5秒 ────→ [收到完整答案]
 *   流式模式：  [发送] → [你] → [好] → [啊] → [!] → [DONE]   每隔几十毫秒收到一个 token
 *
 *   二、SSE 协议基础
 *   SSE 是一个基于 HTTP 的单向服务端推送协议，格式很简单：
 *   data: {"choices":[{"delta":{"content":"你"}}]}
 *   data: {"choices":[{"delta":{"content":"好"}}]}
 *   data: [DONE]
 *
 *   规则：
 *   - 每个事件以 data: 开头
 *   - 事件之间用空行分隔
 *   - : 开头的行是注释（心跳保活）
 *   - data: [DONE] 表示流结束
 *   - 编码必须是 UTF-8
 *   - delta表示的是一个增量数据
 *
 *   和 WebSocket 的区别：
 *   - SSE 是单向（服务端→客户端），WebSocket 是双向
 *   - SSE 基于普通 HTTP，兼容性好，能穿透代理
 *   - SSE 自动重连，WebSocket 需要手动处理
 *
 *   三、代码结构总览
 *   SseStreamClient
 *   ├── StreamCallback 接口        ← 回调三件套：onToken / onComplete / onError
 *   ├── Usage 类                   ← Token 用量统计
 *   ├── streamChat() 核心方法      ← 发请求 + 解析 SSE 流
 *   ├── extractUsage() 工具方法    ← 提取 usage 字段
 *   └── main() 演示
 *
 *   ---
 *   四、回调接口设计
 *   interface StreamCallback {
 *       void onToken(String token);                          // 每个 token 触发一次
 *       void onComplete(String fullContent, Usage usage);    // 流正常结束
 *       void onError(Exception e, String partialContent);    // 发生错误
 *   }
 *
 *   为什么用回调而不是返回值？ — 因为流式数据是异步到达的，不可能用 return 一次性返回。回调是处理异步数据的经典模式。
 *
 *   为什么 onError 要传 partialContent？
 *   网络中断时，用户已经看到了部分文字。把已收到的内容传回去，可以做断点续传或部分展示，而不是全部丢失。
 *
 *   五、请求构建
 *   requestBody.addProperty("stream", true);  // 告诉 API 用流式返回
 *   和普通调用的唯一区别就是 "stream": true。API 收到这个参数后，不再等生成完再返回，而是每生成一个 token 就推送一个 SSE事件。
 *   .addHeader("Accept", "text/event-stream")  // 告诉服务端：我要 SSE 格式
 *   这行不是必须的（大多数 API 看 stream:true 就够了），但加上更规范。
 *
 *   六、OkHttpClient 超时配置
 *   .connectTimeout(30, TimeUnit.SECONDS)   // 建立连接的超时
 *   .readTimeout(60, TimeUnit.SECONDS)      // 两个数据块之间的最大等待时间
 *   .writeTimeout(30, TimeUnit.SECONDS)     // 发送请求的超时
 *   关键：readTimeout 不是整个响应的超时，而是两次读取之间的最大间隔。
 *   - 流式场景下，模型生成一个 token 可能需要 100ms~2s
 *   - 如果 60 秒内没收到任何数据块，才判定超时
 *   - 这和同步调用不同：同步调用的 readTimeout 管的是"从发请求到收完响应"的总时间
 *
 *   七、SSE 流解析核心逻辑：
 *
 *   BufferedReader reader = new BufferedReader(
 *       new InputStreamReader(response.body().byteStream(), StandardCharsets.UTF_8));
 *   为什么用 byteStream() 而不是 string()？ — string() 会等整个响应读完才返回，但 SSE流可能持续几十秒。必须用字节流逐行读取。
 *   为什么显式指定 UTF-8？ — SSE 规范（RFC 8895）要求默认编码是 UTF-8。显式指定避免在不同系统上出现编码问题。
 *
 *
 *   while ((line = reader.readLine()) != null) {
 *   readLine() 会阻塞直到读到一行数据。流结束时返回 null。
 *   Step 1：跳过空行和注释
 *   if (line.isEmpty()) continue;        // 空行 = SSE 事件分隔符
 *   if (line.startsWith(":")) continue;  // 注释行 = 心跳保活
 *   服务端会定期发 :heartbeat 这样的注释行来保持连接不被中间代理断开。这些不是数据，直接跳过。
 *   Step 2：提取 data 字段
 *   if (!line.startsWith("data:")) continue;  // 只处理 data: 行
 *   String data = line.substring(5);           // 去掉 "data:"
 *   if (data.startsWith(" ")) {
 *       data = data.substring(1);              // 去掉一个可选空格 *  SSE 规范规定 data: 后面最多去掉一个空格。data:hello 和 data: hello 都是合法的。
 *   }
 *
 *   Step 3：检查流结束标记
 *   if ("[DONE]".equals(data)) {
 *       streamDone = true;
 *       break;
 *   }
 *   OpenAI 兼容的 API 用 data: [DONE] 标记流结束。这是约定，不是 SSE 协议本身的要求。
 *
 *   Step 4：解析 JSON（加容错）
 *   try {
 *       chunk = JsonParser.parseString(data).getAsJsonObject();
 *   } catch (Exception e) {
 *       System.err.println("JSON 解析失败，跳过: " + data);
 *       continue;  // 跳过这个 chunk，不中断整个流
 *   }
 *   为什么要容错？ — 流式传输中偶尔会收到格式异常的数据。如果因为一个坏 chunk
 *   就中断整个流，用户体验很差。跳过继续处理才是合理的。
 *
 *
 *   Step 5：提取 content
 *   JsonObject delta = choice.getAsJsonObject("delta");
 *   if (delta != null && delta.has("content")) {
 *       JsonElement contentElement = delta.get("content");
 *       if (!contentElement.isJsonNull()) {
 *           String token = contentElement.getAsString();
 *           if (!token.isEmpty()) {
 *               fullContent.append(token);
 *               callback.onToken(token);
 *           }
 *       }
 *   }
 *
 *   流式返回的消息结构和同步不同：
 *   - 同步：choices[0].message.content — 完整内容
 *   - 流式：choices[0].delta.content — 增量内容（每次一个 token）
 *
 *   // 同步返回
 *   {"choices":[{"message":{"content":"你好啊"}}]}
 *
 *   // 流式返回（三个 chunk）
 *   {"choices":[{"delta":{"content":"你"}}]}
 *   {"choices":[{"delta":{"content":"好"}}]}
 *   {"choices":[{"delta":{"content":"啊"}}]}
 *
 *   为什么要多重判空？ — delta 可能为 null、content 可能不存在、content 可能是 JSON null、content
 *   可能是空字符串。每一层都可能出问题，防御性编程是必须的。
 *
 *   Step 6：提取 finish_reason
 *   JsonElement finishElement = choice.get("finish_reason");
 *   if (finishElement != null && !finishElement.isJsonNull()) {
 *       // finish_reason 可能的值：
 *       // - "stop"：正常结束
 *       // - "length"：达到 max_tokens 上限
 *       // - "content_filter"：被安全过滤截断
 *       // - "tool_calls"：模型转入工具调用流程
 *       usage = extractUsage(chunk, usage);
 *   }
 *   注意：这里没有 break！finish_reason 出现不代表流结束，后面可能还有带 usage 的 chunk。真正的结束信号是 data: [DONE]。
 *
 *   ---
 *   八、流结束判断
 *   if (streamDone) {
 *       callback.onComplete(fullContent.toString(), usage);
 *   } else {
 *       callback.onError(
 *           new RuntimeException("SSE 流异常结束：未收到 [DONE] 标记"),
 *           fullContent.toString()
 *       );
 *   }
 *
 *   两种结束方式：
 *   - 正常：收到 data: [DONE]，streamDone=true
 *   - 异常：readLine() 返回 null（连接断开），但没收到 [DONE]
 *
 *   异常情况下，已收到的内容通过 partialContent 传回去，不会丢失。
 *
 *   ---
 *   九、Usage 提取
 *   private static Usage extractUsage(JsonObject chunk, Usage existing) {
 *       if (!chunk.has("usage") || chunk.get("usage").isJsonNull()) {
 *           return existing;  // 这个 chunk 没有 usage，返回之前的
 *       }
 *       // 有 usage 就更新
 *       ...
 *       return usage;
 *   }
 *
 *   为什么要保留 existing？ — 不是每个 chunk 都带 usage。通常只有最后几个 chunk
 *   才有。用"最后一次有效值"的策略，确保最终拿到完整的统计。
 *
 *   ---
 *   十、main 演示中的回调实现
 *
 *   new StreamCallback() {
 *       @Override
 *       public void onToken(String token) {
 *           System.out.print(token);  // 实时打印，不换行
 *       }
 *
 *       @Override
 *       public void onComplete(String fullContent, Usage usage) {
 *           System.out.println("\n--- 流式输出完毕 ---");
 *           // usage 可能为 null，需要判空
 *       }
 *
 *       @Override
 *       public void onError(Exception e, String partialContent) {
 *           // 打印错误 + 已收到的内容
 *       }
 *   }
 *
 *   System.out.print(token) 不换行 — 这就是"打字机效果"的实现。每个 token 到达就立刻打印，用户看到文字逐字出现。
 *
 *   ---
 *   十一、核心思想总结
 *
 *   ┌──────────────────┬──────────────────────────────────────────────────┬──────────────────────────────────────┐
 *   │      设计点      │                       做法                       │                 原因                 │
 *   ├──────────────────┼──────────────────────────────────────────────────┼──────────────────────────────────────┤
 *   │ 回调模式         │ onToken/onComplete/onError                       │ 流式数据异步到达，无法用 return      │
 *   ├──────────────────┼──────────────────────────────────────────────────┼──────────────────────────────────────┤
 *   │ partialContent   │ 错误时传回已收到内容                             │ 网络中断不丢失已展示的文字           │
 *   ├──────────────────┼──────────────────────────────────────────────────┼──────────────────────────────────────┤
 *   │ byteStream()     │ 不用 string()                                    │ SSE 流可能持续几十秒，必须逐行读     │
 *   ├──────────────────┼──────────────────────────────────────────────────┼──────────────────────────────────────┤
 *   │ JSON 解析容错    │ catch 后 continue                                │ 一个坏 chunk 不应中断整个流          │
 *   ├──────────────────┼──────────────────────────────────────────────────┼──────────────────────────────────────┤
 *   │ 多重判空         │ delta != null && has("content") && !isJsonNull() │ API 返回格式不一致，防御性编程       │
 *   ├──────────────────┼──────────────────────────────────────────────────┼──────────────────────────────────────┤
 *   │ readTimeout 含义 │ 两次读取的间隔超时                               │ 流式场景下不是总响应时间             │
 *   ├──────────────────┼──────────────────────────────────────────────────┼──────────────────────────────────────┤
 *   │ [DONE] 结束标记  │ 不靠 finish_reason 判断结束                      │ finish_reason 后可能还有 usage chunk │
 *   └──────────────────┴──────────────────────────────────────────────────┴──────────────────────────────────────┘
 */
public class SseStreamClient {

    private static final String API_URL = Q_LLM_API_URL;
    private static final String API_KEY = ALIBAILIAN_API_KEY;

    // ========== 回调接口 ==========

    /**
     * SSE 流式响应的事件回调
     */
    interface StreamCallback {
        /** 收到一个 content 增量（每个 token 调用一次） */
        void onToken(String token);

        /** 流正常结束，返回完整内容和 Token 统计 */
        void onComplete(String fullContent, Usage usage);

        /** 发生错误，partialContent 是错误发生前已接收到的内容 */
        void onError(Exception e, String partialContent);
    }

    /**
     * Token 用量统计
     */
    static class Usage {
        int promptTokens;
        int completionTokens;
        int totalTokens;

        @Override
        public String toString() {
            return String.format("prompt=%d, completion=%d, total=%d",
                    promptTokens, completionTokens, totalTokens);
        }
    }

    // ========== 核心方法 ==========

    /**
     * 发起流式请求
     *
     * @param model       模型 ID
     * @param systemPrompt System 消息内容
     * @param userMessage  用户消息内容
     * @param callback     事件回调
     */
    public static void streamChat(String model, String systemPrompt,
                                  String userMessage, StreamCallback callback) {
        // 1. 构建请求体
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model", model);
        requestBody.addProperty("temperature", 0.7);
        requestBody.addProperty("max_tokens", 2048);
        requestBody.addProperty("stream", true);

        JsonArray messages = new JsonArray();
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            JsonObject sysMsg = new JsonObject();
            sysMsg.addProperty("role", "system");
            sysMsg.addProperty("content", systemPrompt);
            messages.add(sysMsg);
        }
        JsonObject userMsg = new JsonObject();
        userMsg.addProperty("role", "user");
        userMsg.addProperty("content", userMessage);
        messages.add(userMsg);
        requestBody.add("messages", messages);

        // 2. 创建 HTTP 客户端
        // 关键：readTimeout 是"两个数据块之间的最大等待时间"，不是整个响应的超时
        OkHttpClient client = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)   // 流式场景需要更长
                .writeTimeout(30, TimeUnit.SECONDS)
                .build();

        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .addHeader("Accept", "text/event-stream")  // 明确告诉服务端我要 SSE
                .post(RequestBody.create(requestBody.toString(),
                        MediaType.parse("application/json")))
                .build();

        // 3. 发起请求并解析 SSE 流
        StringBuilder fullContent = new StringBuilder();
        Usage usage = null;

        try (Response response = client.newCall(request).execute()) {
            System.out.println(response);
            System.out.println();
            // 检查 HTTP 状态码
            if (!response.isSuccessful()) {
                String errorBody = response.body() != null ? response.body().string() : "无响应体";
                callback.onError(
                        new RuntimeException("HTTP " + response.code() + ": " + errorBody),
                        fullContent.toString()
                );
                return;
            }

            // 逐行读取 SSE 流（显式指定 UTF-8，SSE 规范要求 UTF-8 编码）
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(response.body().byteStream(), StandardCharsets.UTF_8));
            String line;
            boolean streamDone = false;  // 是否收到了 [DONE] 标记

            while ((line = reader.readLine()) != null) {
                // 跳过空行（SSE 事件分隔符）
                if (line.isEmpty()) {
                    continue;
                }

                // 跳过注释行（心跳保活）
                if (line.startsWith(":")) {
                    continue;
                }

                // 只处理 data: 开头的行（兼容 "data: xxx" 和 "data:xxx" 两种格式）
                if (!line.startsWith("data:")) {
                    continue;
                }

                // 去掉 "data:" 前缀，SSE 标准规定冒号后最多去掉一个可选空格
                String data = line.substring(5);
                if (data.startsWith(" ")) {
                    data = data.substring(1);
                }

                // 检查流结束标记
                if ("[DONE]".equals(data)) {
                    streamDone = true;
                    break;
                }

                // 解析 JSON（加容错）
                JsonObject chunk;
                try {
                    chunk = JsonParser.parseString(data).getAsJsonObject();
                } catch (Exception e) {
                    // JSON 解析失败，跳过这个 chunk，不要中断整个流
                    System.err.println("JSON 解析失败，跳过: " + data);
                    continue;
                }

                // 提取 choices 数组
                JsonArray choices = chunk.getAsJsonArray("choices");
                if (choices == null || choices.isEmpty()) {
                    // 有些平台在最后一个 chunk（stream_options 模式）choices 为空数组
                    // 但可能有 usage 字段
                    usage = extractUsage(chunk, usage);
                    continue;
                }

                JsonObject choice = choices.get(0).getAsJsonObject();

                // 提取 delta 中的 content
                JsonObject delta = choice.getAsJsonObject("delta");
                if (delta != null && delta.has("content")) {
                    JsonElement contentElement = delta.get("content");
                    if (!contentElement.isJsonNull()) {
                        String token = contentElement.getAsString();
                        if (!token.isEmpty()) {
                            fullContent.append(token);
                            callback.onToken(token);
                        }
                    }
                }

                // 提取 finish_reason
                JsonElement finishElement = choice.get("finish_reason");
                if (finishElement != null && !finishElement.isJsonNull()) {
                    String finishReason = finishElement.getAsString();
                    // finish_reason 不只是 "stop"，还可能是：
                    // - "length"：达到 max_tokens 上限，内容被截断
                    // - "content_filter"：被安全过滤截断
                    // - "tool_calls"：模型转入工具调用流程
                    // 这里统一标记为流结束，调用方可根据 finishReason 做更细的处理
                    usage = extractUsage(chunk, usage);
                }
            }

            // 判断流是否正常结束
            if (streamDone) {
                callback.onComplete(fullContent.toString(), usage);
            } else {
                // readLine() 返回 null 但没收到 [DONE]——连接异常关闭
                callback.onError(
                        new RuntimeException("SSE 流异常结束：未收到 [DONE] 标记"),
                        fullContent.toString()
                );
            }

        } catch (Exception e) {
            // 连接异常（超时、网络中断等），把已接收到的内容传给调用方
            callback.onError(e, fullContent.toString());
        }
    }

    /**
     * 从 chunk 中提取 usage 信息
     */
    private static Usage extractUsage(JsonObject chunk, Usage existing) {
        if (!chunk.has("usage") || chunk.get("usage").isJsonNull()) {
            return existing;
        }
        JsonObject usageJson = chunk.getAsJsonObject("usage");
        Usage usage = new Usage();
        usage.promptTokens = usageJson.has("prompt_tokens")
                ? usageJson.get("prompt_tokens").getAsInt() : 0;
        usage.completionTokens = usageJson.has("completion_tokens")
                ? usageJson.get("completion_tokens").getAsInt() : 0;
        usage.totalTokens = usageJson.has("total_tokens")
                ? usageJson.get("total_tokens").getAsInt() : 0;
        return usage;
    }

    // ========== 运行示例 ==========

    public static void main(String[] args) {
        System.out.println("=== SSE 流式调用演示 ===\n");

        streamChat(
            "qwen3-32b",
            "你是一个技术专家，回答简洁清晰。",
            "用两三句话解释一下什么是 SSE 协议？",
            new StreamCallback() {
                @Override
                public void onToken(String token) {
                    // 每收到一个 token 就实时输出（不换行）
                    System.out.print(token);
                }

                @Override
                public void onComplete(String fullContent, Usage usage) {
                    System.out.println("\n");
                    System.out.println("--- 流式输出完毕 ---");
                    System.out.println("完整内容长度：" + fullContent.length() + " 字符");
                    if (usage != null) { // 这里就是生产环境必须要设置的 可能存在返回的data中没有usage字段
                        System.out.println("Token 统计：" + usage);
                    } else {
                        System.out.println("Token 统计：未返回");
                    }
                }

                @Override
                public void onError(Exception e, String partialContent) {
                    System.err.println("\n\n--- 发生错误 ---");
                    System.err.println("错误信息：" + e.getMessage());
                    if (!partialContent.isEmpty()) {
                        System.err.println("已接收到的内容：" + partialContent);
                    }
                }
            }
        );
    }
}
