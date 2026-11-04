
# NavSVLM System Architecture

Dưới đây là sơ đồ kiến trúc và luồng xử lý dữ liệu của **NavSVLM**. STC-Cacher được tích hợp tùy chọn trong Vision Encoder để giảm tính toán lặp giữa các frame. Các giá trị cập nhật 25% patch và chu kỳ tham chiếu 4 frame minh họa cấu hình benchmark; runtime mặc định tắt STC.

```mermaid
flowchart TB
    subgraph INPUTS["1. ĐẦU VÀO (INPUTS)"]
        Prompt["Text Prompt:<br/>'Describe the scene for a visually impaired user...'"]
        InFrame["Khung hình RGB hiện tại<br/>Tiếp nhận tuần tự từ chuỗi ảnh hoặc video"]
    end

    subgraph VISUAL_BRANCH["2. NHÁNH THỊ GIÁC (VISUAL BRANCH)"]

        subgraph VE["Vision Encoder: InternViT + STC-Cacher"]

            Preprocess["Resize 448 × 448 và chuẩn hóa"]
            STCInput["Patch Embedding + CLS + Position Embedding"]
            STCEnabled{"Bật STC-Cacher?"}
            Native["InternViT forward đầy đủ<br/>Không dùng reference cache STC"]

            STCCheck{"Reference Frame?<br/>Chu kỳ N = 4<br/>Frame 1, 5, 9,... hoặc cache không hợp lệ"}

            subgraph CACHER["STC-Cacher — Trong các Encoder Layers"]

                Full["1. Reference<br/>Tính toàn bộ token<br/>Lưu kết quả vào Cache"]

                Cache[("Layer Cache<br/>K, V, Attention, MLP")]

                Select["2. Token Selection<br/>So sánh Cosine Similarity của K<br/>Chọn 25% patch thay đổi nhiều nhất + CLS"]

                Compute["3. Selective Computation<br/>Q/V mới tại token được chọn<br/>Attention trên toàn bộ K hiện tại và V đã ghép<br/>MLP tại token được chọn"]

                Reuse["4. Cache Reuse<br/>Tái sử dụng kết quả token còn lại<br/>Ghép thành đầu ra đầy đủ"]

                Full --> Cache
                Select --> Compute --> Reuse

                Cache -.-> Select
                Cache -.-> Compute
                Cache -.-> Reuse

            end

            STCOutput["Encoder Output<br/>Giữ nguyên số token"]

            STCFeatures["Bỏ CLS + Pixel Shuffle<br/>256 × 4096"]

            Preprocess --> STCInput --> STCEnabled
            STCEnabled -->|Có| STCCheck
            STCEnabled -->|Không| Native
            Native --> STCOutput

            STCCheck -->|YES| Full
            STCCheck -->|NO| Select

            Full --> STCOutput
            Reuse --> STCOutput

            STCOutput --> STCFeatures

        end

        InputProj["Input projection<br/>4096 → 1408"]
        QF["Q-Former (InstructBLIP)<br/>32 query + câu lệnh<br/>Đầu ra 32 × 768"]

        VisTokens["Output projection + Visual MLP<br/>768 → 4096 → 2048<br/>32 token thị giác"]

    end

    subgraph TRACKING_BRANCH["3. NHÁNH THEO DÕI VẬT THỂ (TRACKING BRANCH)"]
        BoTSORT["YOLO11n + BoT-SORT/ReID<br/>Cập nhật tracker qua từng frame<br/>Chọn tối đa 6 vật thể để mã hóa"]

        subgraph TRACKING_ENCODER["Tracking Encoder"]
            AttrsCategorical["Thuộc tính Phân loại:<br/>- Category Label ID<br/>- Relative Position ID"]
            AttrsNumeric["Thuộc tính Số:<br/>- Bounding Box [x1, y1, x2, y2]<br/>- Tốc độ v & Góc chuyển động θ"]
            LabelEmbed["Label & Direction Embeddings"]
            NumericMLP["MLP (Linear -> GELU -> Linear)"]
            Concatenate["Concatenate Features"]
            SlotEmbed["+ Learned Slot Embedding"]
            SetTrans["Set Transformer Encoder<br/>(4 layers Self-Attention + Padding Mask)"]
        end

        TrajTokens["Token thuộc tính vật thể<br/>6 × 384"]
        TrajProj["Trajectory projector + object mask<br/>6 × 2048"]
    end

    subgraph FUSION_MODULE["4. KHÂU DUNG HỢP (TOKEN FUSION)"]
        ConcatHead["Nối theo chiều token sau các phép chiếu riêng<br/>32 + 6 = 38 token, chiều 2048"]
    end

    subgraph LLM_MODULE["5. MÔ HÌNH NGÔN NGỮ & ĐẦU RA"]
        LLM["InternLM2.5 trong InternVL2_5-2B<br/>LoRA adapter"]
        Answer["Đầu ra: Hướng dẫn / Cảnh báo an toàn<br/>(Ví dụ: 'Pedestrian passing at eleven o'clock...')"]
    end

    InFrame --> Preprocess
    STCFeatures --> InputProj --> QF
    QF --> VisTokens

    InFrame --> BoTSORT
    BoTSORT --> AttrsCategorical & AttrsNumeric
    AttrsCategorical --> LabelEmbed
    AttrsNumeric --> NumericMLP
    LabelEmbed --> Concatenate
    NumericMLP --> Concatenate
    Concatenate --> SlotEmbed
    SlotEmbed --> SetTrans
    SetTrans --> TrajTokens

    VisTokens --> ConcatHead
    TrajTokens --> TrajProj --> ConcatHead
    ConcatHead --> LLM

    Prompt --> QF
    Prompt --> LLM
    LLM --> Answer
```

Sơ đồ thể hiện phụ thuộc dữ liệu; runtime thực thi các bước tuần tự. Tracker và mã hóa ảnh cập nhật ở mỗi frame, còn Q-Former, Tracking Encoder, Fusion và LLM chạy khi có yêu cầu sinh hướng dẫn. Tracking Encoder nhận thuộc tính vật thể hiện tại; thông tin thời gian đến từ trạng thái tracker. Thứ tự slot vật thể có thể thay đổi theo từng frame.

Ở selective frame, STC_Cacher so sánh key tại cùng vị trí patch với reference của cùng lớp. Reference không bị ghi đè bởi kết quả selective và được làm mới theo chu kỳ hoặc khi khởi tạo lại trạng thái. Hướng dẫn chi tiết về cơ chế, kích thước tensor và giới hạn thực nghiệm được trình bày trong [báo cáo kiến trúc](BAO_CAO_KIEN_TRUC_NOTION.md).
