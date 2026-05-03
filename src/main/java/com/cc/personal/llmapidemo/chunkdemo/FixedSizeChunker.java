package com.cc.personal.llmapidemo.chunkdemo;

import java.util.ArrayList;
import java.util.List;
public class FixedSizeChunker {
    /**
     * 固定大小分块——最基础的分块方式，按字符数硬切
     */
    public static List<String> chunk(String text, int chunkSize) {
        List<String> chunks = new ArrayList<>();
        int start = 0;
        while (start < text.length()) {
            int end = Math.min(start + chunkSize, text.length());
            chunks.add(text.substring(start, end));
            start = end;
        }
        return chunks;
    }
    public static void main(String[] args) {
        String text = "自签收之日起 7 天内，商品未经使用且不影响二次销售的，"
                + "消费者可申请七天无理由退货。生鲜食品、定制商品、贴身衣物等"
                + "特殊品类不适用此规则，具体以商品详情页标注为准。"
                + "退货运费由消费者承担，如因商品质量问题退货，运费由商家承担。";

        List<String> chunks = chunk(text, 40);

        for (int i = 0; i < chunks.size(); i++) {
            System.out.println("=== 块 " + (i + 1) + " ===");
            System.out.println(chunks.get(i));
            System.out.println();
        }
    }
}
