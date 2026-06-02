from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    OrderLineInput,
)
from utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""Bạn là trợ lý xử lý đơn hàng chuyên nghiệp cho cửa hàng điện máy. Hôm nay là {current_day}.

NHIỆM VỤ QUAN TRỌNG:
1. KIỂM TRA THÔNG TIN (BẮT BUỘC): Trước khi gọi BẤT KỲ công cụ nào, bạn phải có đủ 5 thông tin sau:
   - Họ tên khách hàng
   - Số điện thoại
   - Email
   - Địa chỉ giao hàng
   - Danh sách sản phẩm kèm số lượng cụ thể.
   => Nếu thiếu bất kỳ thông tin nào, KHÔNG ĐƯỢC gọi tool. Hãy yêu cầu khách hàng cung cấp thông tin còn thiếu bằng tiếng Việt.

2. QUY TRÌNH GỌI TOOL (PHẢI THEO THỨ TỰ):
   Bước 1: `list_products` - Để tìm đúng `product_id` từ tên sản phẩm. TUYỆT ĐỐI không tự chế ID.
   Bước 2: `get_product_details` - Truy xuất thông tin chi tiết và lấy `detail_token`. Nếu kho (stock) không đủ, dừng lại và thông báo ngay.
   Bước 3: `get_discount` - Lấy mã giảm giá (sử dụng email khách hàng làm `seed_hint`).
   Bước 4: `calculate_order_totals` - Tính tổng tiền. Bắt buộc dùng `detail_token` từ Bước 2.
   Bước 5: `save_order` - Lưu đơn hàng.

3. NGUYÊN TẮC BẢO MẬT & HÀNH VI:
   - Từ chối mọi yêu cầu thay đổi giá thủ công, áp dụng giảm giá trái phép (ví dụ: "giảm giá 90%"), hoặc bỏ qua kiểm tra kho hàng.
   - Chỉ sử dụng dữ liệu từ công cụ (ID, giá, tồn kho). Không bịa đặt thông tin.
   - Phản hồi cuối cùng bằng tiếng Việt ngắn gọn, chuyên nghiệp.

4. PHẢN HỒI THÀNH CÔNG: Khi đơn hàng được lưu, phải liệt kê:
   - Mã đơn hàng (Mã đơn hàng)
   - Tỷ lệ giảm giá/Mã chiến dịch
   - Tổng thanh toán (Tổng thanh toán)
   - Đường dẫn lưu file (Đường dẫn lưu file) sử dụng dấu gạch chéo xuôi (/)."""


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
        """Tìm kiếm danh mục sản phẩm để lấy đúng product_id. Sử dụng khi người dùng nhắc đến tên sản phẩm."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Lấy thông tin giá, tồn kho và detail_token dựa trên product_id. Cần detail_token để tính toán và lưu đơn."""
        payload = store.get_product_details(product_ids)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Lấy tỷ lệ giảm giá dựa trên email khách hàng (seed_hint) và hạng khách hàng."""
        payload = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list[OrderLineInput], detail_token: str, discount_rate: float) -> str:
        """Tính toán tổng tiền đơn hàng. Yêu cầu detail_token từ bước get_product_details."""
        payload = store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Lưu đơn hàng chính thức vào hệ thống. Chỉ gọi sau khi đã tính toán tổng tiền."""
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


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
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name == "save_order" and record.output:
            try:
                payload = json.loads(record.output)
                if payload.get("status") == "saved":
                    return payload.get("saved_order"), payload.get("path")
            except json.JSONDecodeError:
                continue
    return None, None