# NavVLM System Architecture

Dưới đây là sơ đồ tổng quan về kiến trúc và luồng xử lý dữ liệu của hệ thống **NavVLM**.

```mermaid
flowchart TB
    subgraph INPUTS["1. ĐẦU VÀO (INPUTS)"]
        Prompt["Text Prompt:<br/>'Describe the scene for a visually impaired user...'"]
        InFrame["Khung hình hiện tại (Input Frame)<br/>[1 Ảnh tĩnh]"]
        FrameSeq["Chuỗi 9 khung hình liên tiếp<br/>[Frame Queue]"]
    end

    subgraph VISUAL_BRANCH["2. NHÁNH THỊ GIÁC (VISUAL BRANCH)"]
        VE["Vision Encoder (ViT)<br/>Trích xuất 256 Visual Tokens"]
        QF["Q-Former (InstructBLIP)<br/>32 Learnable Queries + Instruction-Aware Cross-Attention"]
        VisTokens["Compressed Visual Tokens<br/>(Nén 8x: từ 256 còn 32 Tokens)"]
    end

    subgraph TRACKING_BRANCH["3. NHÁNH THEO DÕI VẬT THỂ (TRACKING BRANCH)"]
        BoTSORT["Object Tracker (BoT-SORT + ReID)<br/>Theo dõi tối đa 6 vật thể"]
        
        subgraph TRACKING_ENCODER["Tracking Encoder"]
            AttrsCategorical["Thuộc tính Phân loại:<br/>- Category Label ID<br/>- Relative Position ID"]
            AttrsNumeric["Thuộc tính Số:<br/>- Bounding Box [x1, y1, x2, y2]<br/>- Tốc độ v & Góc chuyển động θ"]
            LabelEmbed["Label & Direction Embeddings"]
            NumericMLP["MLP (Linear -> GELU -> Linear)"]
            Concatenate["Concatenate Features"]
            SlotEmbed["+ Learned Slot Embedding"]
            SetTrans["Set Transformer Encoder<br/>(4 layers Self-Attention + Padding Mask)"]
        end
        
        TrajTokens["Object Trajectory Tokens<br/>(Token Quỹ đạo)"]
    end

    subgraph FUSION_MODULE["4. KHÂU DUNG HỢP (TOKEN FUSION)"]
        ConcatHead["Token Concatenation (ConcatHead)<br/>(Nối trực tiếp các Token)"]
        ProjMLP["Vision-Trajectory Projection MLP"]
    end

    subgraph LLM_MODULE["5. MÔ HÌNH NGÔN NGỮ & ĐẦU RA"]
        LLM["Large Language Model<br/>(InternVL-2B + Q-LoRA Adapter)"]
        Answer["Đầu ra: Hướng dẫn / Cảnh báo an toàn<br/>(Ví dụ: 'Pedestrian passing at eleven o'clock...')"]
    end

    InFrame --> VE
    VE --> QF
    QF --> VisTokens
    FrameSeq --> BoTSORT
    BoTSORT --> AttrsCategorical & AttrsNumeric
    AttrsCategorical --> LabelEmbed
    AttrsNumeric --> NumericMLP
    LabelEmbed --> Concatenate
    NumericMLP --> Concatenate
    Concatenate --> SlotEmbed
    SlotEmbed --> SetTrans
    SetTrans --> TrajTokens
    VisTokens --> ConcatHead
    TrajTokens --> ConcatHead
    ConcatHead --> ProjMLP
    ProjMLP --> LLM
    Prompt --> QF
    Prompt --> LLM
    LLM --> Answer