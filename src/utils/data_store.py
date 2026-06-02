from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from core.schemas import OrderLineInput, ProductRecord


def _normalize(text: str) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    # Giữ lại chữ và số, chuyển về lower case
    compact = re.sub(r"[^a-zA-Z0-9]+", " ", stripped.lower())
    return re.sub(r"\s+", " ", compact).strip()


class OrderDataStore:
    def __init__(self, data_dir: Path, output_dir: Path, *, today: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.today = today or "2026-06-01"
        
        raw_products = json.loads((self.data_dir / "products.json").read_text(encoding="utf-8"))
        self.products = [ProductRecord(**item) for item in raw_products]
        self.product_index = {item.product_id: item for item in self.products}
        
        self.category_aliases = {
            "laptop": "laptop", "notebook": "laptop", "may tinh xach tay": "laptop",
            "monitor": "monitor", "screen": "monitor", "man hinh": "monitor",
            "mouse": "mouse", "chuot": "mouse",
            "keyboard": "keyboard", "ban phim": "keyboard",
            "headphone": "headphone", "tai nghe": "headphone",
            "storage": "storage", "ssd": "storage", "o cung": "storage",
            "phone": "smartphone", "dien thoai": "smartphone",
        }

    @staticmethod
    def build_detail_token(product_ids: list[str]) -> str:
        normalized = "|".join(sorted(product_ids))
        return "DET-" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10].upper()

    def validate_detail_token(self, product_ids: list[str], detail_token: str) -> bool:
        return detail_token == self.build_detail_token(product_ids)

    def canonicalize_category(self, value: str | None) -> str | None:
        if not value:
            return None
        return self.category_aliases.get(_normalize(value), _normalize(value))

    def list_products(
        self,
        *,
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> list[dict]:
        normalized_query = _normalize(query or "")
        # KHÔNG loại bỏ chữ số vì mã model (iPhone 15, M2) rất quan trọng
        query_terms = [t for t in normalized_query.split() if t]
        
        wanted_category = self.canonicalize_category(category)
        wanted_tags = {_normalize(tag) for tag in (required_tags or []) if tag.strip()}
        
        results: list[tuple[int, int, str, dict]] = []

        for product in self.products:
            if in_stock_only and product.stock <= 0:
                continue
            if wanted_category and product.category != wanted_category:
                continue
            if max_unit_price is not None and product.unit_price > max_unit_price:
                continue

            # Tạo tập hợp các trường để tìm kiếm
            name_norm = _normalize(product.name)
            brand_norm = _normalize(product.brand)
            desc_norm = _normalize(product.description)
            tags_norm = [_normalize(t) for t in product.tags]
            
            haystack = f"{name_norm} {brand_norm} {desc_norm} {' '.join(tags_norm)}"
            
            score = 0
            matched_terms = []
            
            # Ưu tiên khớp tên sản phẩm và thương hiệu
            for term in query_terms:
                if term in name_norm:
                    score += 10  # Trọng số cao cho tên
                    matched_terms.append(term)
                elif term in brand_norm:
                    score += 5
                    matched_terms.append(term)
                elif term in haystack:
                    score += 2
                    matched_terms.append(term)

            for tag in wanted_tags:
                if tag in tags_norm:
                    score += 5
                    matched_terms.append(tag)

            if query_terms and not matched_terms:
                continue

            results.append(
                (
                    score,
                    product.stock,
                    product.product_id,
                    {
                        "product_id": product.product_id,
                        "name": product.name,
                        "brand": product.brand,
                        "unit_price": product.unit_price,
                        "stock": product.stock,
                        "category": product.category,
                    },
                )
            )

        # Sắp xếp theo score giảm dần, sau đó đến tồn kho
        results.sort(key=lambda x: (-x[0], -x[1]))
        return [item[-1] for item in results[:limit]]

    def get_product_details(self, product_ids: list[str]) -> dict:
        details: list[dict] = []
        valid_ids = []
        for pid in product_ids:
            product = self.product_index.get(pid)
            if product:
                valid_ids.append(pid)
                details.append({
                    "status": "ok",
                    "product_id": product.product_id,
                    "name": product.name,
                    "unit_price": product.unit_price,
                    "stock": product.stock,
                    "warranty_months": product.warranty_months
                })
            else:
                details.append({"product_id": pid, "status": "not_found"})
        
        return {
            "status": "ok" if valid_ids else "error",
            "detail_token": self.build_detail_token(valid_ids) if valid_ids else "",
            "items": details,
        }

    def get_discount(self, *, seed_hint: str, customer_tier: str = "standard") -> dict:
        normalized_seed = seed_hint.strip().lower()
        digest = hashlib.sha256(f"{customer_tier}|{normalized_seed}".encode("utf-8")).hexdigest()
        discount_rate = 0.2 if int(digest[-2:], 16) % 10 < 4 else 0.1
        return {
            "status": "ok",
            "discount_rate": discount_rate,
            "campaign_code": f"FLASH-{int(discount_rate * 100):02d}",
        }

    def calculate_order_totals(self, *, items: list[OrderLineInput], detail_token: str, discount_rate: float) -> dict:
        product_ids = [item.product_id for item in items]
        if not self.validate_detail_token(product_ids, detail_token):
            return {"status": "error", "errors": ["Invalid detail token. Please call get_product_details first."]}

        subtotal = 0
        lines = []
        for item in items:
            product = self.product_index.get(item.product_id)
            if not product or item.quantity > product.stock:
                return {"status": "error", "errors": [f"Stock issue with {item.product_id}"]}
            
            line_total = product.unit_price * item.quantity
            subtotal += line_total
            lines.append({
                "product_id": product.product_id,
                "name": product.name,
                "quantity": item.quantity,
                "unit_price": product.unit_price,
                "line_total": line_total
            })

        discount_amount = int(subtotal * discount_rate)
        return {
            "status": "ok",
            "items": lines,
            "pricing": {
                "subtotal": subtotal,
                "discount_rate": discount_rate,
                "discount_amount": discount_amount,
                "final_total": subtotal - discount_amount,
            }
        }

    def save_order(self, **kwargs) -> dict:
        # Tái sử dụng logic tính toán để đảm bảo tính chính xác
        pricing = self.calculate_order_totals(
            items=kwargs['items'], 
            detail_token=kwargs['detail_token'], 
            discount_rate=kwargs['discount_rate']
        )
        if pricing["status"] != "ok": return pricing

        # Tạo Order ID duy nhất
        seed = f"{kwargs['customer_email']}-{kwargs['customer_phone']}-{self.today}"
        order_id = "ORD-" + hashlib.sha1(seed.encode()).hexdigest()[:10].upper()
        
        payload = {
            "order_id": order_id,
            "customer": {
                "name": kwargs['customer_name'],
                "phone": kwargs['customer_phone'],
                "email": kwargs['customer_email'],
                "address": kwargs['shipping_address']
            },
            "items": pricing["items"],
            "pricing": pricing["pricing"],
            "campaign": kwargs['campaign_code'],
            "path": f"artifacts/orders/{order_id}.json"
        }
        
        dest = self.output_dir / f"{order_id}.json"
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        
        return {"status": "saved", "order_id": order_id, "path": payload["path"], "saved_order": payload}