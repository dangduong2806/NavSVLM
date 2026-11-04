# NavSVLM — Hệ thống hỗ trợ mô tả và định hướng di chuyển bằng mô hình thị giác–ngôn ngữ

**Tích hợp thông tin theo dõi vật thể và cơ chế STC-Cacher để tối ưu quá trình suy luận**

| Thông tin | Nội dung |
|---|---|
| Tên project | NavSVLM |
| Sinh viên thực hiện | [Điền họ và tên, mã sinh viên] |
| Giảng viên hướng dẫn | [Điền họ và tên] |
| Đơn vị / học phần | [Điền khoa, bộ môn hoặc học phần] |
| Ngày cập nhật báo cáo | 28/10/2026 |
| Phạm vi báo cáo | Kiến trúc đã triển khai, quy trình suy luận và kết quả benchmark được lưu trong project |

## 1. Tóm tắt đề tài

NavSVLM là một hệ thống thử nghiệm hỗ trợ người khiếm thị tiếp nhận thông tin về môi trường xung quanh thông qua hướng dẫn bằng ngôn ngữ tự nhiên. Hệ thống kết hợp đặc trưng hình ảnh với thông tin vị trí và chuyển động của vật thể, từ đó sinh mô tả tập trung vào chướng ngại vật, hướng tương đối và hành động cần chú ý.

Kiến trúc gồm hai nhánh xử lý chính. Nhánh thị giác sử dụng InternViT của mô hình `OpenGVLab/InternVL2_5-2B`, kết hợp Q-Former để rút gọn biểu diễn ảnh. Nhánh theo dõi sử dụng YOLO11n và BoT-SORT với cấu hình ReID, sau đó mã hóa thuộc tính của tối đa sáu vật thể bằng Tracking Encoder. Hai nhánh được chiếu về cùng không gian biểu diễn, nối thành chuỗi token và đưa vào mô hình ngôn ngữ có LoRA adapter để sinh hướng dẫn.

Project tích hợp STC-Cacher vào Vision Encoder nhằm khai thác sự tương đồng giữa các khung hình liên tiếp. Thay vì tính lại toàn bộ phép toán cho mọi token tại mọi khung hình, cơ chế này chọn một phần token cần cập nhật và tái sử dụng kết quả từ khung hình tham chiếu cho phần còn lại.

Theo các tệp benchmark đã lưu, trên một chuỗi 9 khung hình chạy bằng CPU, cấu hình STC cập nhật 25% patch token và làm mới cache mỗi 4 khung hình đạt mức tăng tốc quan sát được **1,885 lần đối với mã hóa ảnh** và **2,200 lần đối với toàn backend**. Đây là kết quả thực nghiệm ban đầu trên phạm vi nhỏ; chưa đủ để kết luận về khả năng thời gian thực, độ chính xác điều hướng hoặc hiệu quả trên các môi trường triển khai khác.

## 2. Bài toán và động cơ thực hiện

Đầu ra phục vụ hỗ trợ di chuyển cần truyền đạt được thông tin có ý nghĩa đối với người dùng: vật thể nào đang ở phía trước, vị trí tương đối của vật thể và tình huống nào cần chú ý. Một mô tả chỉ dựa trên ảnh hiện tại có thể thiếu thông tin về chuyển động. Ngược lại, dữ liệu tracking gồm nhãn và tọa độ chưa trực tiếp tạo thành một chỉ dẫn dễ hiểu.

Project đặt ra bài toán kết hợp hai loại thông tin này trong cùng một quá trình sinh ngôn ngữ. Đồng thời, việc xử lý video kéo dài đòi hỏi giảm chi phí tính toán lặp lại giữa các khung hình tương tự nhau.

**Đầu vào:** chuỗi ảnh RGB theo thứ tự thời gian, video hoặc camera qua giao diện dòng lệnh, cùng câu lệnh yêu cầu mô tả cảnh.

**Đầu ra:** hướng dẫn bằng ngôn ngữ tự nhiên dựa trên khung hình gần nhất và trạng thái tracking tích lũy. Giao diện web có thể đọc văn bản bằng chức năng tổng hợp giọng nói của trình duyệt.

**Phạm vi hiện tại:** hệ thống hỗ trợ nhận biết và mô tả tình huống; chưa triển khai định vị bản đồ, lập kế hoạch đường đi toàn cục hoặc điều khiển thiết bị di chuyển.

## 3. Mục tiêu và nội dung triển khai

| Mục tiêu | Nội dung trong project |
|---|---|
| Kết hợp thông tin ảnh và chuyển động | Dung hợp token ảnh với token thuộc tính vật thể được theo dõi |
| Giảm số token ảnh đưa vào mô hình ngôn ngữ | Q-Former rút gọn 256 token sau pixel shuffle thành 32 query token |
| Giảm tính toán lặp lại giữa các khung hình | Tích hợp STC-Cacher trực tiếp vào các lớp InternViT |
| Duy trì xử lý chuỗi | Giữ mô hình trong bộ nhớ, duy trì tracker và cache giữa các frame |
| Minh họa khả năng sử dụng | Giao diện web phát lại chuỗi ảnh mẫu, sinh hướng dẫn và đọc thành tiếng |
| Đánh giá hiệu năng có thể kiểm tra lại | Hai script benchmark, báo cáo JSON chi tiết và Markdown tổng hợp |

Đóng góp ở cấp project tập trung vào tích hợp các thành phần, điều chỉnh STC cho cấu trúc InternViT, quản lý trạng thái suy luận và xây dựng quy trình đánh giá. Các mô hình nền, Q-Former và thuật toán tracking là những thành phần được tái sử dụng; không xem đây là các mô hình được huấn luyện từ đầu trong phạm vi báo cáo này.

## 4. Kiến trúc hệ thống

### 4.1. Luồng xử lý tổng quát

```text
Khung hình RGB theo thời gian
        │
        ├── Resize + chuẩn hóa → InternViT [STC tùy chọn]
        │                         → Pixel shuffle: 256 token ảnh
        │                         → Q-Former + prompt: 32 token
        │                         → Các lớp chiếu + visual MLP
        │                         → [1, 32, 2048]
        │
        └── YOLO11n → BoT-SORT + ReID
                    → Thuộc tính tối đa 6 vật thể
                    → Tracking Encoder + projector
                    → [1, 6, 2048]
                                  │
                 Nối hai nhánh: [1, 38, 2048]
                                  │
                    Prompt + LLM có LoRA adapter
                                  │
                      Hướng dẫn bằng văn bản
                                  │
                  Đọc thành tiếng trên giao diện web
```

Sơ đồ mô tả luồng phụ thuộc dữ liệu. Trong runtime hiện tại, các bước được thực hiện tuần tự; hình vẽ hai nhánh không có nghĩa là hai nhánh chạy song song.

### 4.2. Nhánh thị giác và Q-Former

Mỗi ảnh được chuyển về kích thước **448 × 448**, chuẩn hóa rồi đưa qua InternViT. Với đường xử lý một ảnh đơn hiện tại, đặc trưng sau pixel shuffle có kích thước `[1, 256, 4096]`.

Q-Former lấy nguồn từ `Salesforce/instructblip-flan-t5-xl`, sử dụng **32 query token** và có tiếp nhận prompt. Nhờ đó, bước rút gọn biểu diễn ảnh được điều kiện hóa theo yêu cầu mô tả cảnh. Các lớp chiếu kết nối đặc trưng InternViT với Q-Former và đưa đầu ra trở lại visual MLP của mô hình nền. Kết quả cuối của nhánh ảnh là `[1, 32, 2048]`.

Việc giảm từ 256 xuống 32 là **giảm 8 lần số token ảnh tại bước này**, không phải tuyên bố tăng tốc toàn hệ thống 8 lần. Khi thêm sáu vị trí token tracking, đầu vào đa phương thức có 38 token, chưa tính token văn bản của prompt.

### 4.3. Nhánh phát hiện và theo dõi vật thể

Runtime sử dụng trọng số `yolo11n.pt` để phát hiện vật thể và BoT-SORT với `with_reid=True`, `model="auto"` để duy trì trạng thái theo dõi. Tối đa sáu đối tượng được đưa vào bộ mã hóa ở mỗi thời điểm.

Mỗi vật thể được biểu diễn bằng:

| Nhóm thuộc tính | Nội dung |
|---|---|
| Nhãn đối tượng | ID ánh xạ từ tên lớp sang bộ từ vựng của checkpoint |
| Vị trí tương đối | Một trong các hướng 10 giờ, 11 giờ, 12 giờ, 1 giờ và 2 giờ |
| Hộp giới hạn | Bốn tọa độ được chuẩn hóa theo chiều rộng và chiều cao ảnh |
| Chuyển động | Góc chuyển động và độ lớn vận tốc được suy ra từ trạng thái tracker |
| Mặt nạ hợp lệ | Phân biệt đối tượng có thật với các vị trí đệm |

Sáu đối tượng được chọn theo tọa độ cạnh dưới của bounding box giảm dần. Đây là quy tắc ưu tiên theo vị trí trên ảnh, chưa phải phép đo khoảng cách vật lý hoặc điểm rủi ro va chạm. Tốc độ được chuẩn hóa theo kích thước ảnh; không diễn giải trực tiếp thành đơn vị m/s.

Bộ từ vựng chứa 107 nhãn và một ID đệm/không xác định. Kích thước bộ từ vựng không đồng nghĩa YOLO11n hiện nhận biết đầy đủ 107 lớp đó. Runtime ánh xạ tên lớp detector vào bộ từ vựng; các tên không có ánh xạ sử dụng ID không xác định.

### 4.4. Tracking Encoder và dung hợp

Tracking Encoder kết hợp embedding của nhãn, embedding hướng và đặc trưng số qua MLP, bổ sung slot embedding rồi xử lý bằng Transformer Encoder. Cấu hình được nạp trong `main.py` gồm **4 lớp, 4 attention head và chiều đặc trưng 384**. Projector chuyển sáu token tracking về chiều 2048; các slot không hợp lệ được đưa về vector 0.

Thông tin thời gian chủ yếu đến từ trạng thái tracker qua các frame. Tracking Encoder xử lý tập thuộc tính vật thể ở thời điểm hiện tại; nó không nhận trực tiếp một tensor chứa toàn bộ lịch sử chín khung hình.

Hai nhánh được **chiếu riêng về chiều 2048 trước khi nối**:

```text
32 token ảnh + 6 token tracking = 38 token đa phương thức
```

Các token này thay thế những vị trí `<IMG_CONTEXT>` trong embedding đầu vào của mô hình ngôn ngữ. Cách dung hợp đang chạy là nối token; không có một khối cross-attention riêng giữa hai nhánh trước LLM.

### 4.5. Mô hình ngôn ngữ và đầu ra

Runtime nạp mô hình nền `OpenGVLab/InternVL2_5-2B` và LoRA adapter trong checkpoint `intern-qformer-concat-1807-epoch3`. Cấu hình adapter ghi nhận rank `r=16`, `lora_alpha=32`, `lora_dropout=0.05`.

Prompt yêu cầu mô tả dựa trên khung hình cuối, chú ý chướng ngại vật gần, hướng di chuyển và hành động của người dùng. Cấu hình sinh sử dụng `do_sample=False`, `max_new_tokens=128` và KV cache của LLM.

Runtime hiện tại không bật lượng tử hóa 4-bit khi nạp mô hình. Vì vậy, báo cáo mô tả đây là **suy luận với LoRA adapter**. Chưa có đủ hồ sơ huấn luyện được đối chiếu để khẳng định phương pháp huấn luyện là QLoRA, số epoch thực tế hoặc cấu hình tối ưu hóa chỉ từ tên thư mục checkpoint.

## 5. Cơ chế tối ưu STC-Cacher

### 5.1. Nguyên lý triển khai

STC-Cacher trong project được điều chỉnh từ thuật toán selective-key của mã nguồn STC đi kèm, bổ sung khả năng tương thích với fused QKV, Q/K normalization và LayerScale của InternViT. Adapter chỉ áp dụng trong suy luận và không sửa trọng số checkpoint.

Tại **khung hình tham chiếu**, hệ thống chạy đầy đủ các token và lưu key, value, kết quả attention và kết quả MLP của từng lớp.

Tại **khung hình cập nhật chọn lọc**, mỗi lớp thực hiện:

1. Tính key hiện tại cho toàn bộ token.
2. So sánh cosine similarity giữa key của từng patch và key tại khung hình tham chiếu.
3. Chọn các patch có độ tương đồng thấp nhất; luôn bổ sung CLS token.
4. Tính lại query/value, đầu ra attention và MLP cho các token được chọn; attention vẫn sử dụng key của toàn bộ token.
5. Ghép kết quả vừa tính với value, đầu ra attention và MLP được tái sử dụng từ tham chiếu.

Đầu ra giữ nguyên số token để các thành phần phía sau tiếp tục sử dụng được. Việc chọn token dựa trên thay đổi của key thị giác; chưa sử dụng bounding box hoặc điểm rủi ro từ nhánh tracking.

### 5.2. Cấu hình thực nghiệm

| Tham số | Giá trị |
|---|---:|
| Tỷ lệ patch được cập nhật | 0,25 |
| Chu kỳ làm mới cache | 4 khung hình |
| Reference trong chuỗi 9 frame, đếm từ 1 | 1, 5, 9 |
| Selective trong chuỗi 9 frame | 2, 3, 4, 6, 7, 8 |
| Token MLP mỗi lớp ở reference | 1.025 |
| Token MLP mỗi lớp ở selective | 257 = 256 patch + 1 CLS |

Tỷ lệ 25% áp dụng cho patch token trên selective frame. Vì vẫn có reference frame, phép tính key toàn phần, chọn token và sao chép cache, không thể suy ra hệ thống giảm 75% tổng chi phí hoặc tăng tốc 4 lần.

Qua chín frame, số lượt token đi qua MLP của toàn bộ các lớp giảm từ **221.400 xuống 110.808**, tức khoảng **49,95%**, theo báo cáo profiling. Đây là số lượt xử lý token của MLP, không phải phép đo FLOPs toàn mô hình.

### 5.3. Quản lý trạng thái

Mô hình được nạp một lần trong mỗi worker; tracker và STC cache được duy trì giữa các khung hình. Khi đổi stream, thay đổi kích thước nguồn hoặc phát lại một chuỗi, runtime xóa trạng thái thời gian nhưng giữ trọng số trong bộ nhớ.

Ứng dụng cần gọi reset khi cắt cảnh hoặc seek; hiện chưa có bộ phát hiện cắt cảnh tự động. Adapter sử dụng trạng thái của một stream, do đó các yêu cầu được tuần tự hóa để tránh trộn cache giữa các nguồn video.

## 6. Dữ liệu và tài nguyên sử dụng

Mẫu benchmark nằm trong `wad_sample/images`, gồm **9 ảnh** từ `00000000.jpg` đến `00000008.jpg`. Theo `wad_sample/metadata.json`, nguồn mẫu được ghi là `minhdang0901/WAD_Images_All_Size`, thuộc cùng một thư mục frame. Các tệp được đưa vào mô hình theo thứ tự tên đã sắp xếp.

Ba lần lặp benchmark sử dụng lại cùng chuỗi ảnh này. Vì vậy, có 27 lượt xử lý frame cho mỗi chế độ nhưng chỉ có **9 frame đầu vào khác nhau**, không phải 27 mẫu độc lập.

| Thành phần | Tài nguyên được nạp |
|---|---|
| Mô hình thị giác–ngôn ngữ nền | `OpenGVLab/InternVL2_5-2B` |
| Nguồn Q-Former | `Salesforce/instructblip-flan-t5-xl` |
| Adapter và các nhánh bổ sung | `intern-qformer-concat-1807-epoch3` |
| Detector | `yolo11n.pt` |
| Bộ từ vựng nhãn | `Tracking_Encoder/label_vocabulary.json` |
| Báo cáo benchmark ảnh | `outputs/stc_benchmark.json` và `.md` |
| Báo cáo benchmark backend | `outputs/system_benchmark.json` và `.md` |

Báo cáo này chưa xác minh quy mô tập huấn luyện, cách chia train/validation/test, tính độc lập của mẫu benchmark với tập huấn luyện hoặc kết quả trên toàn bộ dữ liệu WAD. Những thông tin đó cần bổ sung từ hồ sơ huấn luyện trước khi đánh giá khả năng khái quát hóa.

## 7. Chức năng đã triển khai

| Chức năng | Trạng thái và phạm vi |
|---|---|
| Xử lý thư mục ảnh | Phát lại chuỗi theo thứ tự và sinh hướng dẫn tại frame cuối |
| Xử lý video/camera | Có qua CLI; xử lý tuần tự và sinh hướng dẫn theo chu kỳ |
| Quản lý mô hình | Tái sử dụng mô hình đã nạp trong cùng process |
| STC | Tùy chọn, bật bằng cấu hình môi trường; mặc định runtime tắt |
| Giao diện web | Xem chuỗi ảnh mẫu, sinh kết quả, theo dõi tiến độ và log |
| Đọc hướng dẫn | Dùng giọng đọc có sẵn trong trình duyệt |
| Benchmark | Đo riêng mã hóa ảnh và toàn backend |

Giao diện web hiện là giao diện phát lại mẫu, chưa phải giao diện camera trực tiếp. Đầu ra mẫu đang sử dụng tiếng Anh; chưa có kết quả đánh giá đầu ra tiếng Việt. Khả năng đọc thành tiếng phụ thuộc hỗ trợ và giọng đọc của trình duyệt.

## 8. Benchmark và kết quả thực nghiệm

### 8.1. Mục tiêu và nguồn số liệu

Benchmark được thiết kế để trả lời ba câu hỏi:

1. Đường tính chọn lọc có tương thích số học với forward gốc khi cập nhật 100% token hay không?
2. STC cải thiện tốc độ mã hóa ảnh bao nhiêu, đồng thời làm thay đổi đặc trưng đầu ra ở mức nào?
3. Khi tích hợp đầy đủ tracking, Q-Former và LLM, thời gian hoàn thành một chuỗi thay đổi ra sao?

**Toàn bộ kết quả dưới đây được tổng hợp từ các tệp JSON đã lưu có trạng thái `complete`; không phải một lượt benchmark mới chạy khi biên soạn báo cáo.** Hash của bốn tệp mã nguồn được ghi trong benchmark hệ thống đã được đối chiếu và khớp với phiên bản hiện tại. Hash chín ảnh trong benchmark STC cũng khớp với dữ liệu hiện có.

### 8.2. Thiết lập thực nghiệm

| Yếu tố | Cấu hình đã ghi nhận |
|---|---|
| Hệ điều hành | Windows 11, chuỗi platform `Windows-11-10.0.26200-SP0` |
| Thiết bị tính toán | CPU |
| PyTorch | `2.9.0+cpu` |
| Kiểu số | FP32 |
| Số luồng PyTorch | 4 |
| Attention của benchmark vision | Eager; TF32 tắt |
| Đầu vào vision | Một ảnh 448 × 448 mỗi lượt |
| Số ảnh / số lần lặp | 9 ảnh / 3 lần lặp mỗi chế độ |
| Warmup vision | 2 lượt mỗi phép đo chuỗi, không tính vào thời gian |
| Warmup backend | 2 frame mỗi chế độ; có làm nóng bước sinh hướng dẫn |
| STC | Tỷ lệ 0,25; chu kỳ cache 4 |
| Lịch sinh hướng dẫn của benchmark backend | Mỗi 9 frame và frame cuối; mẫu này có 1 câu trả lời/lượt |
| Sinh văn bản | Không lấy mẫu ngẫu nhiên; tối đa 128 token mới |

Tệp báo cáo chưa lưu tên CPU, tổng RAM, chế độ nguồn điện, tải nền và nhiệt độ máy. Những thông tin này cần được ghi thêm trong đợt đo tiếp theo. Không thay thế chúng bằng cấu hình máy tại một thời điểm khác rồi coi là cấu hình lịch sử của phép đo.

### 8.3. Phạm vi đo và nguyên tắc so sánh

| Nội dung | `benchmark_stc.py` | `benchmark_system.py` |
|---|---|---|
| Phần được đo | InternViT + pixel shuffle | Đọc JPG đến khi có văn bản đầu ra |
| Giải mã ảnh, tiền xử lý, truyền dữ liệu | Loại khỏi thời gian chính | Có tính |
| YOLO và tracking | Không chạy | Có tính trên mọi frame |
| Q-Former, Tracking Encoder, LLM | Không chạy | Có tính khi sinh hướng dẫn |
| Nạp mô hình và warmup | Không tính | Không tính; thời gian nạp được ghi riêng |
| HTTP, trình duyệt, giọng đọc | Không tính | Không tính |
| Profiling chi tiết | Lượt riêng theo lớp | Lượt riêng theo giai đoạn |

Các chế độ trong từng benchmark dùng chung trọng số và đầu vào; trạng thái cache được reset trước chuỗi đo. Benchmark backend reset cả tracker và giữ lịch sinh hướng dẫn giống nhau. Thứ tự đo được luân phiên giữa các lần lặp để giảm ảnh hưởng của thứ tự chạy.

Baseline của benchmark hệ thống vẫn xử lý tracking và vision trên **mọi frame**. Nó chỉ tắt STC, không phải phương án chỉ chạy Vision Encoder một lần ở ảnh cuối.

Với cùng lượng công việc:

```text
Speedup = Tổng thời gian baseline / Tổng thời gian STC
Mức giảm thời gian (%) = (1 − Tổng thời gian STC / Tổng thời gian baseline) × 100
Throughput = Tổng số frame xử lý / Tổng thời gian tương ứng
```

Speedup lớn hơn 1 biểu thị nhanh hơn. FPS trong benchmark ảnh chỉ phản ánh tốc độ mã hóa; FPS của backend đã bao gồm công việc sinh hướng dẫn theo lịch cấu hình.

### 8.4. Kiểm tra tính đúng đắn của đường cập nhật 100%

Trước khi đo tốc độ, `benchmark_stc.py` chạy chế độ `selective_100`: các frame giữa hai lần làm mới vẫn đi qua đường tính chọn lọc, nhưng cập nhật toàn bộ token. Đầu ra được so sánh với forward gốc dùng cùng trọng số, tại các hidden state, đầu ra vision và đặc trưng sau pixel shuffle.

| Kiểm tra | Kết quả |
|---|---|
| Sai số cho phép | `atol=1e-4`, `rtol=1e-4` |
| Số selective frame được kiểm tra | 6 |
| Kiểm tra đường MLP thực sự nhận số token dự kiến | Có |
| Kết quả tổng hợp | **PASS** |

Kết quả xác nhận đường cập nhật 100% vượt qua kiểm tra tương đương số học trên mẫu và cấu hình này. Nó không chứng minh rằng chế độ cập nhật 25% giữ nguyên đặc trưng hoặc giữ nguyên chất lượng hướng dẫn.

### 8.5. Kết quả benchmark Vision Encoder

| Chế độ | TB/chuỗi 9 frame (s) | TB/frame (ms) | P95/frame (ms) | Encoding FPS | Speedup | Cache STC (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 48,913 | 5.434,83 | 7.402,90 | 0,184 | 1,000× | 0 |
| Selective 100% | 39,837 | 4.426,28 | 6.892,61 | 0,226 | 1,228× | 384,375 |
| STC 25% | 25,943 | 2.882,58 | 7.030,35 | 0,347 | **1,885×** | 384,375 |

STC 25% giảm **46,96%** thời gian mã hóa ảnh theo tổng thời gian ba lượt đo. Trong chế độ này, reference frame trung bình mất **5.093,89 ms**, còn selective frame mất **1.776,93 ms**. P95 vẫn cao vì chuỗi còn các frame tính đầy đủ và thời gian chạy có biến động.

Kết quả từng lần đo, đơn vị giây cho một chuỗi 9 frame:

| Lần đo | Baseline | Selective 100% | STC 25% |
|---|---:|---:|---:|
| 1 | 63,650 | 34,683 | 20,938 |
| 2 | 51,080 | 52,330 | 37,878 |
| 3 | 32,010 | 32,496 | 19,014 |

Các lượt đo vision dao động đáng kể. Selective 100% cũng có thời gian trung bình thấp hơn baseline dù không giảm số token MLP; vì vậy không nên quy toàn bộ khác biệt thời gian giữa các đường chạy cho việc bỏ bớt token. Cần thêm lượt đo trong điều kiện tải máy ổn định để tách ảnh hưởng của cách thực thi và biến động môi trường.

### 8.6. Sai lệch đặc trưng và chi phí bộ nhớ

| Chỉ số ở selective frame với STC 25% | Kết quả |
|---|---:|
| Cosine similarity trung bình của đặc trưng sau pixel shuffle | 0,840605 |
| Cosine similarity nhỏ nhất | 0,805528 |
| Kiểm tra tensor đầu ra hữu hạn và đúng kích thước | Đạt |
| Kiểm tra tương đương số học với dung sai `1e-4` | Không đạt (`quality.passed=false`) |
| Dung lượng tensor cache tham chiếu | 384,375 MiB |

Chế độ STC 25% là phép tính xấp xỉ: đặc trưng đã thay đổi so với baseline. Cosine similarity không phải độ chính xác điều hướng; giá trị 0,840605 không có nghĩa hệ thống đạt 84,06% độ chính xác.

384,375 MiB chỉ là dung lượng tensor tham chiếu STC. Benchmark chưa đo đỉnh RAM toàn process trên CPU. Vì chạy CPU, trường bộ nhớ CUDA để trống; không diễn giải thành hệ thống không có chi phí bộ nhớ.

### 8.7. Profiling riêng theo lớp

| Chế độ | Thời gian MLP (s) | Phần còn lại trong lớp (s) | Lượt token qua MLP trên toàn bộ lớp |
|---|---:|---:|---:|
| Baseline | 15,4871 | 15,2630 | 221.400 |
| Selective 100% | 15,5296 | 16,3686 | 221.400 |
| STC 25% | 7,9754 | 10,0827 | 110.808 |

Trong lượt profiling riêng, MLP đạt mức tăng tốc khoảng **1,942 lần** với STC 25%. Phần còn lại gồm attention, projection, chọn token, sao chép cache, normalization, residual và overhead của hook; không gọi toàn bộ phần đó là “thời gian attention”. Các số liệu có gắn đo đạc này không dùng thay cho kết quả tốc độ chính ở mục 8.5.

### 8.8. Kết quả benchmark toàn backend

| Chỉ số | Baseline | STC 25% |
|---|---:|---:|
| Thời gian TB/chuỗi 9 frame | 42,183 s | **19,173 s** |
| Độ lệch chuẩn giữa ba lượt | 12,569 s | 0,253 s |
| Thời gian nhỏ nhất | 28,714 s | 18,886 s |
| Thời gian lớn nhất | 53,599 s | 19,366 s |
| Throughput | 0,213 frame/s | 0,469 frame/s |
| Speedup | 1,000× | **2,200×** |
| Thời gian TB của frame có sinh hướng dẫn | 13.404,78 ms | 7.775,51 ms |
| Thời gian TB của frame không sinh hướng dẫn | 3.597,20 ms | 1.424,61 ms |
| Số token trả về trung bình/câu | 20 | 20 |
| Cache STC cuối chuỗi | 0 MiB | 384,375 MiB |

Thời gian backend giảm quan sát được **54,55%**. Do chỉ sinh một câu tại frame cuối, thời gian từ lúc bắt đầu chuỗi tới hướng dẫn hoàn chỉnh gần bằng thời gian toàn chuỗi: 42,183 s với baseline và 19,173 s với STC. Đây không phải time-to-first-token.

Thời gian nạp mô hình/runtime được ghi riêng là khoảng **21,79 s**, chưa nằm trong các giá trị trên. Bước khởi tạo trễ của YOLO và các kernel suy luận đầu tiên được bao phủ bởi warmup bị loại khỏi phép đo, nên không xem 21,79 s là toàn bộ chi phí khởi động lạnh đến khi có câu trả lời đầu tiên.

| Lần đo | Baseline (s) | STC (s) |
|---|---:|---:|
| 1 | 28,714 | 19,366 |
| 2 | 44,235 | 19,265 |
| 3 | 53,599 | 18,886 |

Baseline dao động lớn và có xu hướng tăng qua ba lượt. Chưa có dữ liệu tải CPU hoặc nhiệt độ để xác định nguyên nhân. Do đó, 2,200× được báo cáo là **mức tăng tốc quan sát được trong đợt đo**, chưa phải một hệ số tăng tốc ổn định đã xác nhận trên nhiều lần thử độc lập.

Hai benchmark được chạy độc lập. Không lấy thời gian vision ở mục 8.5 trừ khỏi thời gian backend ở đây để suy ra thời gian LLM hoặc các thành phần khác. Tương tự, việc speedup backend lớn hơn speedup vision không chứng minh STC trực tiếp tăng tốc LLM.

### 8.9. Phân tích các giai đoạn của backend

Lượt profiling riêng cho kết quả sau:

| Giai đoạn | Baseline (s) | STC (s) |
|---|---:|---:|
| Đọc JPG và giải mã RGB | 0,0224 | 0,0298 |
| YOLO, tracking và tạo thuộc tính | 1,0786 | 1,2263 |
| Tiền xử lý ảnh vision | 0,0995 | 0,1039 |
| InternViT + pixel shuffle | 50,5004 | 32,8600 |
| Q-Former và các lớp chiếu thị giác | 0,3530 | 0,1331 |
| Tracking Encoder và projector | 0,0085 | 0,0047 |
| Dung hợp và chuẩn bị prompt | 0,0020 | 0,0013 |
| LLM generation: prefill + decode | 9,9569 | 5,0403 |
| Giải mã token thành văn bản | 0,0008 | 0,0005 |

Trong tổng thời gian các giai đoạn của lượt diagnostic baseline, vision chiếm khoảng **81,4%**, còn LLM generation chiếm khoảng **16,1%**. Điều này cho thấy tối ưu Vision Encoder là một hướng đáng khảo sát trong cấu hình CPU hiện tại.

STC chỉ trực tiếp thay đổi bước vision. Chênh lệch thời gian ở Q-Former hoặc LLM chưa đủ để quy là tác dụng trực tiếp của cache; lượt diagnostic này đều trả về 20 token nhưng các giai đoạn vẫn có thể chịu biến động môi trường thực thi. Prefill và decode chưa được đo riêng.

Các con số trên thuộc một lượt có instrumentation sau các lần đo chính. Không cộng chúng rồi coi là thời gian trung bình backend ở mục 8.8, và không tính tỷ trọng của chúng trên mẫu số của một lượt đo khác.

### 8.10. Đối chiếu văn bản đầu ra

Baseline và STC cho văn bản giống nhau trong **3/3 lần đối chiếu**:

> at 11 o'clock direction, there are pedestrians passing by, pay attention to avoid.

Diễn giải tiếng Việt: “Ở hướng 11 giờ có người đi bộ đang đi qua, hãy chú ý tránh.” Đây là bản dịch để trình bày, không phải đầu ra tiếng Việt đã được mô hình kiểm thử.

Kết quả 3/3 là mức khớp văn bản trên ba lần chạy lại cùng một mẫu. Nó không phải độ chính xác 100% trên ba tình huống độc lập. Hơn nữa, cả ba câu trả lời đều được sinh tại frame thứ 9 — một **reference frame**. Do đó, phép thử hiện tại chưa kiểm tra ảnh hưởng của đặc trưng xấp xỉ lên hướng dẫn được sinh ở selective frame.

Để khảo sát trường hợp này, có thể chạy cùng chuỗi với `--guidance-every 4`; cả baseline và STC sẽ sinh câu tại frame 4, 8 và 9. Đây là cấu hình đánh giá bổ sung, chưa có số liệu trong các bảng kết quả hiện tại.

## 9. Đánh giá kết quả và hạn chế

### 9.1. Kết quả đạt được

Project đã xây dựng được một luồng suy luận thống nhất từ chuỗi ảnh đến văn bản hướng dẫn, tích hợp thông tin tracking và cơ chế cache thị giác. Mô hình có thể được tái sử dụng trong một worker, tránh nạp lại cho từng yêu cầu. Hai bộ benchmark giúp kiểm tra riêng tính tương thích của STC, hiệu năng mã hóa ảnh và thời gian xử lý backend.

Kết quả đã lưu cho thấy tiềm năng giảm thời gian xử lý trên mẫu CPU đang xét, đồng thời định lượng được chi phí bộ nhớ cache và sai lệch đặc trưng. Đây là cơ sở để thiết kế đợt đánh giá rộng hơn.

### 9.2. Các giới hạn cần nêu khi trình bày

| Giới hạn | Ảnh hưởng đến kết luận |
|---|---|
| Một chuỗi 9 frame | Chưa đại diện cho nhiều cảnh, điều kiện sáng hoặc mức chuyển động |
| Chỉ ba lượt đo | Chưa ước lượng được độ ổn định và khoảng tin cậy đáng tin cậy |
| Baseline có biến động lớn | Cần kiểm soát môi trường trước khi khẳng định hệ số tăng tốc |
| Chỉ có kết quả CPU FP32 | Chưa suy rộng sang GPU, điện thoại hoặc thiết bị nhúng |
| Chưa đo peak RAM CPU | Chưa kết luận mức bộ nhớ tối thiểu để triển khai |
| Chưa có nhãn chuẩn về hướng dẫn | Chưa tính được độ chính xác, tỷ lệ bỏ sót hoặc cảnh báo sai |
| Chỉ sinh câu trên reference frame trong báo cáo hiện có | Chưa đánh giá ngôn ngữ đầu ra khi sử dụng đặc trưng selective |
| STC làm thay đổi đặc trưng | Cần đo đánh đổi giữa tốc độ và chất lượng ở tác vụ cuối |
| Throughput backend dưới 1 frame/s trong phép đo | Chưa đạt cơ sở để tuyên bố xử lý thời gian thực |
| Chưa có đối chứng bỏ tracking hoặc bỏ Q-Former | Chưa định lượng riêng đóng góp của hai thành phần |
| Chưa đo HTTP, giao diện và phát âm thanh | Chưa có độ trễ đầy đủ từ camera đến lời nói |

Với phạm vi đánh giá hiện tại, sản phẩm phù hợp cho minh họa và nghiên cứu. Chưa có dữ liệu kiểm chứng để kết luận hệ thống đủ độ tin cậy cho việc đưa ra quyết định di chuyển trong môi trường thực.

## 10. Hướng phát triển và kế hoạch đánh giá tiếp theo

Các nội dung dưới đây là **kế hoạch đề xuất**, chưa phải kết quả đã thực hiện.

| Hướng thực hiện | Thiết kế đánh giá đề xuất | Kết quả cần thu thập |
|---|---|---|
| Mở rộng dữ liệu | Nhiều chuỗi độc lập; cảnh tĩnh, chuyển động nhanh, đông người và thay đổi ánh sáng | Kết quả theo từng nhóm cảnh và tổng hợp theo chuỗi |
| Ổn định phép đo | Ghi CPU/RAM, nguồn điện, tải nền; làm nóng ổn định; tăng số lượt và cân bằng thứ tự | Trung bình, median, P95, độ lệch chuẩn và khoảng tin cậy theo đơn vị mẫu phù hợp |
| Khảo sát tham số STC | Tỷ lệ cập nhật 0,125; 0,25; 0,5 và nhiều chu kỳ cache | Đường đánh đổi tốc độ, sai lệch đặc trưng và chất lượng hướng dẫn |
| Sinh câu trên selective frame | Đo cùng lịch sinh câu cho baseline và STC, ví dụ chu kỳ 4 | Sai khác văn bản và đánh giá nội dung tại frame 4, 8, 9 |
| Đối chứng thành phần | So sánh có/không tracking, có/không Q-Former với đường chạy và trọng số tương thích | Đóng góp riêng của từng thành phần |
| Đánh giá nội dung | Xây dựng nhãn chuẩn và tiêu chí: đúng vật thể, đúng hướng, phát hiện nguy cơ, hành động phù hợp, ngắn gọn | Điểm từ người chấm, tỷ lệ bỏ sót nguy cơ và cảnh báo sai |
| Đánh giá triển khai | Chạy CPU/GPU với dtype được ghi rõ, đo RAM/VRAM và toàn tuyến camera–âm thanh | Độ trễ thực tế và yêu cầu tài nguyên |
| Phát triển chức năng | Phát hiện cắt cảnh, chọn token theo vùng vật thể, đánh giá đầu ra tiếng Việt | So sánh chất lượng và độ trễ trước/sau |

Nếu dùng đánh giá thủ công, nên ẩn tên cấu hình khi chấm và áp dụng cùng bộ tiêu chí cho các chế độ. Đánh giá ngữ nghĩa cần dựa trên cảnh và nhãn chuẩn; khớp nguyên văn với baseline chỉ là một chỉ số đối chiếu phụ.

## 11. Kết luận

NavSVLM đã hình thành một hệ thống thử nghiệm kết hợp biểu diễn thị giác, thuộc tính vật thể theo dõi và mô hình ngôn ngữ để sinh hướng dẫn mô tả môi trường. Cơ chế STC-Cacher được tích hợp vào InternViT nhằm giảm tính toán lặp lại giữa các khung hình, đồng thời giữ nguyên giao diện đặc trưng cho những thành phần tiếp theo.

Trên dữ liệu benchmark đã lưu, cấu hình STC 25% với chu kỳ 4 đạt mức tăng tốc quan sát được 1,885 lần cho mã hóa ảnh và 2,200 lần cho toàn backend CPU. Kết quả đi kèm sai lệch đặc trưng và khoảng 384,375 MiB cache tham chiếu. Đường cập nhật 100% vượt qua kiểm tra số học, còn chất lượng hướng dẫn tại selective frame và hiệu quả trên tập dữ liệu rộng vẫn cần được kiểm chứng.

Hướng phát triển ưu tiên là mở rộng thực nghiệm, kiểm soát biến động phép đo và đánh giá chất lượng hướng dẫn theo nhãn chuẩn. Những bước này sẽ giúp xác định rõ mức đánh đổi giữa tốc độ, tài nguyên và độ tin cậy của hệ thống.

## Phụ lục A. Lệnh tái lập benchmark

Các lệnh dưới đây chạy từ thư mục `NavSVLM`, trong môi trường Python đã cài dependency và có đủ model/checkpoint trong cache. Đóng worker demo trước khi benchmark để hạn chế tranh chấp tài nguyên. Lệnh dùng tên báo cáo mới để giữ lại số liệu đã lưu.

**Đo Vision Encoder và kiểm tra selective 100%:**

```powershell
python benchmark_stc.py --images-dir wad_sample/images --device cpu --dtype float32 --limit 9 --ratio 0.25 --cache-interval 4 --repeats 3 --warmup 2 --threads 4 --profile-layers --output outputs/stc_benchmark_recheck.json
```

**Đo toàn backend theo cấu hình đã báo cáo:**

```powershell
python benchmark_system.py --images-dir wad_sample/images --device cpu --dtype float32 --limit 9 --guidance-every 9 --ratio 0.25 --cache-interval 4 --repeats 3 --warmup-frames 2 --threads 4 --seed 0 --profile-stages --output outputs/system_benchmark_recheck.json
```

**Đánh giá bổ sung khi sinh câu ở selective frame:**

```powershell
python benchmark_system.py --images-dir wad_sample/images --device cpu --dtype float32 --limit 9 --guidance-every 4 --ratio 0.25 --cache-interval 4 --repeats 3 --warmup-frames 2 --threads 4 --seed 0 --profile-stages --output outputs/system_selective_guidance.json
```

Hai script mặc định sử dụng cache Hugging Face offline; thêm `--online` nếu cần và cho phép tải model. Mỗi lệnh xuất JSON chi tiết và tệp Markdown cùng tên. So sánh baseline/STC bên trong cùng cấu hình; không coi chu kỳ sinh câu 4 và 9 là cùng khối lượng công việc.

## Phụ lục B. Chạy demo

Từ thư mục `NavSVLM`, với môi trường đã chạy được pipeline:

```powershell
$env:STC_PATCH_VISION="1"
$env:STC_UPDATE_TOKEN_RATIO="0.25"
$env:STC_CACHE_INTERVAL="4"
python Frontend/server.py
```

Mở `http://127.0.0.1:8000`, xem chuỗi ảnh và chọn **Generate guidance**. Đợi văn bản đầu ra, sau đó có thể đọc lại hoặc sao chép kết quả. Worker giữ mô hình trong bộ nhớ cho các yêu cầu tiếp theo; khởi động lại launcher khi thay đổi cấu hình STC.

## Phụ lục C. Nguồn đối chiếu trong project

| Nguồn | Nội dung làm căn cứ |
|---|---|
| `main.py` | Model nền, detector, prompt, tải checkpoint, chiều token, thứ tự chiếu và dung hợp, cấu hình sinh |
| `streaming_pipeline.py` | Xử lý từng frame, preprocessing, reset trạng thái và vòng đời mô hình |
| `Vision_Encoder/stc_cacher.py` | Thuật toán chọn token, cache và chu kỳ tham chiếu |
| `Concat_Q_Former/qformer_bridge.py` | Các thành phần kết nối Q-Former |
| `Tracking_Encoder/trajectory_branch.py` | Embedding thuộc tính, Transformer Encoder, mask và projector |
| `intern-qformer-concat-1807-epoch3/qformer_bridge_config.json` | Nguồn Q-Former, số query và cấu hình prompt-aware |
| `intern-qformer-concat-1807-epoch3/adapter_config.json` | Cấu hình LoRA được lưu |
| `wad_sample/metadata.json` | Nguồn và danh sách frame của mẫu |
| `benchmark_stc.py`, `outputs/stc_benchmark.json` | Phương pháp đo vision, kiểm tra số học, thời gian, sai lệch đặc trưng và cache |
| `benchmark_system.py`, `outputs/system_benchmark.json` | Phạm vi backend, lịch sinh câu, thời gian và văn bản đối chiếu |
| `pipeline_timing.py` | Cơ chế profiling tách biệt |
| `Frontend/README.md` | Phạm vi và cách vận hành demo |

Tên tệp trong phụ lục được ghi tương đối với thư mục `NavSVLM`. Báo cáo dựa vào mã thực thi và kết quả JSON; sơ đồ khái quát cũ trong `system.md` không được dùng để thay thế các chi tiết triển khai đã đối chiếu ở trên.
