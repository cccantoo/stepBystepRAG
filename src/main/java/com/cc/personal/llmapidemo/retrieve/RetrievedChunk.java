package com.cc.personal.llmapidemo.retrieve;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * 检索到的 chunk，包含内容和元数据
 */
@Getter
@NoArgsConstructor
@AllArgsConstructor
public class RetrievedChunk {
    private String content;      // chunk 文本内容
    private String source;       // 来源文档名
    private String sourceUrl;    // 原文链接
    private String updateTime;   // 更新时间
    private Double score;        // 重排序得分
}
