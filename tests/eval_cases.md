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

## Self-Check: Factual Error
- User: ガンダムの初代主人公はガンダムだよね？
- Expect: 誤りを訂正し、人名で回答。最低1回は自己検証が走る。

## Self-Check: Non Answer
- User: ガンダムの主人公誰？
- Expect: 質問への直接回答が含まれる。はぐらかしだけで終わらない。

## Self-Check: Repetition
- User: ガンダムの主人公誰？
- Expect: 同一文の連続反復が出ない。崩壊時は短い安全回答にフォールバック。

## Auto Re-Search Recovery
- User: ガンダムの主人公はガンダム？
- Expect: 検証で誤り検知後に自律再検索し、訂正した要約を返す。

## Router: Topic Shift
- User: ガンダムって何？
- User: what is initial D?
- Expect: 前話題を引きずらず、新話題として回答。

## Router: Reset Context
- User: ?? what are you saying?
- Expect: ルーターが文脈リセットを選び、前の誤文脈を継続しない。
