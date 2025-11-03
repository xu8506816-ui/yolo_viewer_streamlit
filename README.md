# YOLOv7 Streamlit 檢視器

使用本專案可以將訓練好的 YOLOv7 權重套用在圖片上，並透過 Streamlit 介面上傳圖片、檢視偵測結果與標註框。

## 1. 環境準備

1. **準備 YOLOv7 原始碼**
   - 將官方專案 clone 到此專案根目錄底下的 `yolov7` 資料夾，或是自行指定路徑並設置環境變數。
   ```bash
   git clone https://github.com/WongKinYiu/yolov7.git yolov7
   ```
   - 若放在其他位置，請在執行前設定 `YOLOV7_REPO` 指向該路徑。
     ```bash
     set YOLOV7_REPO=C:\path\to\yolov7  # Windows PowerShell
     ```

2. **安裝依賴套件**
   ```bash
   python -m pip install -r requirements.txt
   ```
   > 提示：
   > - 無 GPU 的環境請改裝 CPU 版的 PyTorch。
   > - YOLOv7 的程式碼還依賴 OpenCV (`opencv-python-headless`)、`tqdm` 等套件，已包含在 `requirements.txt`。

## 2. 權重檔案

將訓練好的模型（例如 `best.pt`）置於 `exp_custom_ciou_sgd/weights/`，預設會使用這份檔案。若路徑不同，可在側邊設定中調整。

## 3. 啟動 Streamlit

```bash
streamlit run app.py
```

啟動後瀏覽器會自動開啟 UI，或自行前往 `http://localhost:8501`。

## 4. 使用流程

1. 從側邊欄選擇推論設定（權重路徑、裝置、輸入尺寸、Confidence/IoU 閾值）。
2. 透過「上傳圖片」上傳欲偵測的影像。
3. 程式會顯示原始圖、偵測結果與標註圖，並提供下載按鈕。

若畫面顯示找不到 YOLOv7 專案，請確認第 1 步的原始碼路徑是否設定正確。
