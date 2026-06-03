from __future__ import annotations

import json
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import build_chat_model, normalize_content
from core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    date_str = today or "unknown"
    return f"""Bạn là trợ lý đặt hàng điện tử chuyên nghiệp. Hôm nay là {date_str}.

=== KIỂM TRA TRƯỚC KHI GỌI TOOL ===
Trước khi gọi BẤT KỲ tool nào, hãy xác nhận đầy đủ 5 thông tin sau từ khách hàng:
1. Tên khách hàng (customer_name)
2. Số điện thoại (customer_phone)
3. Email (customer_email)
4. Địa chỉ giao hàng (shipping_address)
5. Ít nhất một sản phẩm cụ thể kèm số lượng mặt hàng

Nếu THIẾU BẤT KỲ thông tin nào ở trên → Hãy yêu cầu khách hàng cung cấp thông tin còn thiếu, sau đó DỪNG lại và TUYỆT ĐỐI KHÔNG gọi bất kỳ một tool nào.

=== QUY TẮC TỪ CHỐI (KHÔNG gọi tool, trả lời thẳng) ===
Từ chối ngay lập tức và không gọi bất kỳ tool nào nếu người dùng yêu cầu:
- Bỏ qua, gian lận hoặc ghi đè kiểm tra tồn kho thực tế.
- Áp dụng giảm giá giả / giảm giá thủ công không thông qua hệ thống.
- Tạo hóa đơn giả, đơn hàng giả hoặc thông tin không có thực.
- Bỏ qua catalog sản phẩm hoặc các chính sách bán hàng của công ty.
- Bất kỳ yêu cầu gian lận hoặc vi phạm chính sách bảo mật/hệ thống nào khác.

=== TRÌNH TỰ TOOL BẮT BUỘC ===
Khi đã nhận đủ 5 thông tin yêu cầu, bắt buộc phải thực hiện các tool theo đúng trình tự sau:
1. `list_products` – Tìm kiếm sản phẩm phù hợp trong hệ thống.
2. `get_product_details` – Lấy thông tin chi tiết và `detail_token` xác thực.
3. `get_discount` – Lấy thông tin chương trình giảm giá (luôn truyền seed_hint = email của khách hàng).
4. `calculate_order_totals` – Tính toán tổng tiền đơn hàng (Sử dụng chính xác `detail_token` lấy từ bước 2).
5. `save_order` – Lưu đơn hàng vào hệ thống (Sử dụng đầy đủ thông tin khách hàng và kết quả từ các bước trước).

=== NGUYÊN TẮC GROUNDING ===
- CHỈ sử dụng product_id, giá tiền, và số lượng tồn kho thực tế từ kết quả tool trả về, KHÔNG tự ý bịa đặt.
- CHỈ dùng `detail_token` được trả về từ `get_product_details`.
- CHỈ dùng `discount_rate` và `campaign_code` từ kết quả của `get_discount`.
- CHỈ dùng `final_total` từ kết quả của `calculate_order_totals`.
- CHỈ dùng đường dẫn `save_path` được trả về từ kết quả của `save_order`.

=== ĐẦU RA ===
Trả lời ngắn gọn, súc tích bằng tiếng Việt. Nếu đơn hàng thành công, bắt buộc phải bao gồm: mã đơn hàng (order_id), tổng tiền đơn hàng (final_total), và đường dẫn file lưu đơn hàng (save_path).
"""


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        result = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        result = store.get_product_details(product_ids)
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        result = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        from core.schemas import OrderLineInput
        parsed_items = [OrderLineInput(**i) if isinstance(i, dict) else i for i in items]
        result = store.calculate_order_totals(
            items=parsed_items,
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        from core.schemas import OrderLineInput
        parsed_items = [OrderLineInput(**i) if isinstance(i, dict) else i for i in items]
        result = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=parsed_items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    data_dir = data_dir or DEFAULT_DATA_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    store = OrderDataStore(data_dir=data_dir, output_dir=output_dir, today=today)
    model = build_chat_model(provider=provider, model_name=model_name)
    tools = build_tools(store)
    system_prompt = build_system_prompt(today=today)

    return create_agent(model=model, tools=tools, system_prompt=system_prompt)


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )

    result = agent.invoke({"messages": [("user", query)]})
    messages = result.get("messages", [])

    final_answer = extract_final_answer(messages)
    tool_calls = extract_tool_calls(messages)
    saved_order, save_path = extract_saved_order(tool_calls)

    return AgentResult(
        answer=final_answer,
        tool_calls=tool_calls,
        saved_order=saved_order,
        save_path=save_path,
    )


def extract_final_answer(messages) -> str:
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = normalize_content(msg.content)
            if content and content.strip():
                answer = content.strip()
                break
    return answer


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    records: list[ToolCallRecord] = []
    call_map: dict[str, dict] = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                call_map[tc["id"]] = {"name": tc["name"], "args": tc["args"]}
        elif isinstance(msg, ToolMessage):
            call_info = call_map.get(msg.tool_call_id, {})
            try:
                output = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                output = msg.content
            records.append(
                ToolCallRecord(
                    tool=call_info.get("name", "unknown"),
                    args=call_info.get("args", {}),
                    output=output,
                )
            )
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for tc in reversed(tool_calls):
        if tc.tool == "save_order":
            output = tc.output
            if isinstance(output, dict) and output.get("status") == "ok":
                return output.get("order"), output.get("save_path")
    return None, None