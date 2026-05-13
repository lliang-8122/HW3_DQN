# HW3-1 學習理解報告：Naive DQN 與 Experience Replay 的比較 (Static 模式)

## 實驗目標與概述
本報告總結了深度 Q 網絡 (Deep Q-Network, DQN) 在 `Gridworld` 環境的 `static` (完全靜態) 模式下的實作與表現。實驗目標是訓練一個智能體 (Agent) 走到目標並避開陷阱，我們透過比較「不包含任何技巧的 Naive DQN」與「加入經驗回放緩衝區 (Experience Replay Buffer) 的 DQN」，來觀察並分析 Experience Replay 對於神經網路訓練的影響。

## 1. 實作內容說明 (Naive DQN)
在 Naive DQN 的實作中，智能體與環境進行逐步互動，過程中**完全沒有使用**經驗回放緩衝區。

**訓練流程**：
1. **狀態觀測**：智能體觀測當前狀態 (將 $4 \times 4$ 網格攤平為 64 維度的特徵陣列)。
2. **選擇動作**：使用 $\epsilon$-greedy 策略選擇動作。
3. **環境互動**：執行動作後，環境回傳下一個狀態以及獎勵。
4. **即時網路更新**：直接使用這「單一步驟」的轉換資料 `(state, action, reward, next_state)` 來立即更新 Q-Network。

**觀察結果與限制分析**：
由於連續時間步產生的訓練樣本具有極高的時間相關性，這違反了隨機梯度下降 (SGD) 所需的「獨立同分布 (i.i.d)」假設。神經網路很容易對最近走過的狀態序列產生過度擬合 (Overfitting) 並發生「災難性遺忘」。在訓練初期的 Loss 曲線中，可以明顯看到激烈的震盪波動。但由於環境是 Static 模式 (起點目標固定)，最終網路還是能硬生生背出這條路徑而收斂。

## 2. 引入經驗回放 (Experience Replay Buffer)
為了解決 Naive DQN 震盪與樣本相關性的問題，我們在程式中加入了 Experience Replay Buffer (使用 `collections.deque` 實作)。

**運作機制**：
- 智能體不再每走一步就拿當下的資料訓練，而是將每次的轉換 `(state, action, reward, next_state, done)` 存入 Buffer 中（容量設定為 1000）。
- 當 Buffer 內累積到足夠數量的資料（大於 `batch_size=32`）時，我們會在每一步從中**隨機抽取 (random sample)** 一個 Mini-batch 的歷史資料來進行梯度更新。

**帶來的好處與觀察**：
1. **打破資料相關性 (Decorrelation)**：由於資料是從歷史中隨機抽取的，Mini-batch 內的樣本之間不再具有強烈的連續性，滿足了 SGD 的前提假設，大幅降低了更新方向的偏誤。
2. **提升資料使用效率**：在 Naive DQN 中，每筆轉換資料用過一次就丟棄；而在 Experience Replay 中，一筆資料（例如偶然走到終點獲得的高額獎勵）可以被重複抽樣多次，讓神經網路能更有效率地學習稀有且有價值的經驗。
3. **Loss 曲線更為平滑**：對比兩張實驗結果圖（`hw3_1_naive_results.png` 與 `hw3_1_er_results.png`），可以明顯觀察到加入 Experience Replay 後的 Loss 下降曲線變得更加平滑且穩定，早期的震盪幅度也大幅縮小，收斂速度與品質都獲得了提升。

## 結論
這項實驗明確展示了 Experience Replay 是深度強化學習中不可或缺的機制。即便在簡單的 Static 環境下，它也能有效地去相關化並平滑化資料分佈，讓 DQN 的學習過程從劇烈波動轉為穩定收斂。
