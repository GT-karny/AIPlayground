# Evaluation Cases

## Context Resolution
- User: ガンダムって何？
- User: 主人公誰？
- Expect: 初代作品の主人公としてアムロ・レイに言及。曖昧確認は不要。

## Ambiguous Follow-up
- User: iPhone 16の価格教えて
- User: で、安いのは？
- Expect: 価格軸で比較回答。文脈が切れずに回答。

## Clarification Needed
- User: これどう思う？
- Expect: 2-3択の確認質問を返す。

## Repetition Guard
- User: ガンダムの主人公誰？
- Expect: 同一文の反復がない。最大7文程度で収まる。

## Strict Factual
- User: 今日のドル円レートは？
- Expect: factual_strict で検索を使い、URL付き回答または保留。

## Casual Chat
- User: 今日は疲れた
- Expect: chat モードで自然な会話。検索しない。
