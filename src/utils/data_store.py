from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from core.schemas import OrderLineInput, ProductRecord

CATEGORY_ALIASES: dict[str, str] = {
    "notebook": "laptop",
    "notebooks": "laptop",
    "laptops": "laptop",
    "phone": "smartphone",
    "phones": "smartphone",
    "smartphones": "smartphone",
    "mobile": "smartphone",
    "mobiles": "smartphone",
    "tablet": "tablet",
    "tablets": "tablet",
    "headphone": "headphones",
    "earphone": "headphones",
    "earphones": "headphones",
    "earbuds": "headphones",
    "watch": "smartwatch",
    "watches": "smartwatch",
    "smartwatches": "smartwatch",
    "monitor": "monitor",
    "monitors": "monitor",
    "screen": "monitor",
    "screens": "monitor",
    "keyboard": "keyboard",
    "keyboards": "keyboard",
    "mouse": "mouse",
    "mice": "mouse",
    "camera": "camera",
    "cameras": "camera",
    "printer": "printer",
    "printers": "printer",
    "speaker": "speaker",
    "speakers": "speaker",
    "router": "router",
    "routers": "router",
    "hard drive": "storage",
    "hard disk": "storage",
    "ssd": "storage",
    "hdd": "storage",
}


class OrderDataStore:
    """
    Student TODO:
    - Load `products.json`.
    - Build lookup helpers for product IDs and normalized search.
    - Save final orders under `artifacts/orders/`.
    """

    def __init__(self, data_dir: Path, output_dir: Path, *, today: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.today = today
        self.output_dir.mkdir(parents=True, exist_ok=True)

        products_path = self.data_dir / "products.json"
        raw = json.loads(products_path.read_text(encoding="utf-8"))
        self.products: list[ProductRecord] = [ProductRecord(**p) for p in raw]
        self.product_index: dict[str, ProductRecord] = {p.product_id: p for p in self.products}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_detail_token(product_ids: list[str]) -> str:
        key = ",".join(sorted(product_ids))
        digest = hashlib.sha1(key.encode()).hexdigest()[:10]
        return f"DET-{digest.upper()}"

    def validate_detail_token(self, product_ids: list[str], token: str) -> bool:
        return self.build_detail_token(product_ids) == token

    def canonicalize_category(self, raw: str) -> str:
        normalized = raw.strip().lower()
        return CATEGORY_ALIASES.get(normalized, normalized)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

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
        """
        Student TODO:
        - Search by product name, brand, category, tags, and description.
        - Return compact catalog summaries that the model can reuse in later tool calls.
        """
        results: list[tuple[int, ProductRecord]] = []
        canon_cat = self.canonicalize_category(category) if category else None
        query_tokens = re.split(r"\s+", query.lower()) if query else []

        for p in self.products:
            if in_stock_only and p.stock <= 0:
                continue
            if canon_cat and self.canonicalize_category(p.category) != canon_cat:
                continue
            if max_unit_price is not None and p.unit_price > max_unit_price:
                continue
            if required_tags:
                p_tags = [t.lower() for t in (p.tags or [])]
                if not all(t.lower() in p_tags for t in required_tags):
                    continue

            score = 0
            if query_tokens:
                searchable = " ".join([
                    p.name, p.brand, p.category,
                    p.description or "",
                    " ".join(p.tags or []),
                ]).lower()
                for tok in query_tokens:
                    if tok in searchable:
                        score += 1
                if score == 0:
                    continue

            results.append((score, p))

        results.sort(key=lambda x: (-x[0], x[1].name))
        return [
            {
                "product_id": p.product_id,
                "name": p.name,
                "brand": p.brand,
                "category": p.category,
                "unit_price": p.unit_price,
                "stock": p.stock,
                "tags": p.tags,
            }
            for _, p in results[:limit]
        ]

    def get_product_details(self, product_ids: list[str]) -> dict:
        """
        Student TODO:
        - Return exact pricing, stock, category, and warranty information for each product ID.
        - Return a deterministic validation token that later tools can verify.
        - Preserve the input order or document how you reorder it.
        """
        items = []
        missing = []
        for pid in product_ids:
            p = self.product_index.get(pid)
            if p is None:
                missing.append(pid)
            else:
                items.append({
                    "product_id": p.product_id,
                    "name": p.name,
                    "brand": p.brand,
                    "category": p.category,
                    "unit_price": p.unit_price,
                    "stock": p.stock,
                    "tags": p.tags,
                    "description": p.description,
                    "warranty_months": p.warranty_months,
                })
        if missing:
            return {"status": "error", "missing_ids": missing}
        detail_token = self.build_detail_token(product_ids)
        return {"status": "ok", "detail_token": detail_token, "items": items}

    def get_discount(self, *, seed_hint: str, customer_tier: str = "standard") -> dict:
        """
        Student TODO:
        - Simulate a random campaign discount with deterministic seeding for grading.
        - Supported discount rates should be `0.1` or `0.2`.
        """
        seed = f"{seed_hint}:{customer_tier}"
        digest = hashlib.sha256(seed.encode()).hexdigest()
        val = int(digest[:8], 16)
        discount_rate = 0.2 if val % 2 == 0 else 0.1
        campaign_code = f"CAMP-{digest[:6].upper()}"
        return {
            "discount_rate": discount_rate,
            "campaign_code": campaign_code,
            "customer_tier": customer_tier,
        }

    def calculate_order_totals(self, *, items: list[OrderLineInput], detail_token: str, discount_rate: float) -> dict:
        """
        Student TODO:
        - Validate product IDs.
        - Validate the detail token produced by `get_product_details(...)`.
        - Validate requested quantities against stock.
        - Compute subtotal, discount amount, and final total.
        - Return an error payload instead of throwing for common user mistakes.
        """
        product_ids = [item.product_id for item in items]

        if not self.validate_detail_token(product_ids, detail_token):
            return {"status": "error", "reason": "invalid_detail_token"}

        line_items = []
        for item in items:
            p = self.product_index.get(item.product_id)
            if p is None:
                return {"status": "error", "reason": f"unknown_product:{item.product_id}"}
            if p.stock < item.quantity:
                return {
                    "status": "error",
                    "reason": "insufficient_stock",
                    "product_id": item.product_id,
                    "requested": item.quantity,
                    "available": p.stock,
                }
            line_total = p.unit_price * item.quantity
            line_items.append({
                "product_id": p.product_id,
                "name": p.name,
                "unit_price": p.unit_price,
                "quantity": item.quantity,
                "line_total": line_total,
            })

        subtotal = sum(li["line_total"] for li in line_items)
        discount_amount = round(subtotal * discount_rate, 2)
        final_total = round(subtotal - discount_amount, 2)

        return {
            "status": "ok",
            "line_items": line_items,
            "subtotal": subtotal,
            "discount_rate": discount_rate,
            "discount_amount": discount_amount,
            "final_total": final_total,
        }

    def save_order(
        self,
        *,
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
    ) -> dict:
        """
        Student TODO:
        - Recompute totals before saving.
        - Build a deterministic order ID.
        - Persist the final JSON payload to the output directory.
        - Return both the saved order payload and the saved file path.
        """
        totals = self.calculate_order_totals(
            items=items, detail_token=detail_token, discount_rate=discount_rate
        )
        if totals.get("status") != "ok":
            return totals

        items_key = ",".join(sorted(f"{i.product_id}:{i.quantity}" for i in items))
        id_seed = f"{customer_email}:{customer_phone}:{items_key}"
        id_digest = hashlib.sha256(id_seed.encode()).hexdigest()[:12].upper()
        order_id = f"ORD-{id_digest}"

        payload = {
            "order_id": order_id,
            "customer_name": customer_name,
            "customer_phone": customer_phone,
            "customer_email": customer_email,
            "shipping_address": shipping_address,
            "customer_tier": customer_tier,
            "campaign_code": campaign_code,
            "discount_rate": discount_rate,
            "notes": notes,
            "line_items": totals["line_items"],
            "subtotal": totals["subtotal"],
            "discount_amount": totals["discount_amount"],
            "final_total": totals["final_total"],
        }

        out_path = self.output_dir / f"{order_id}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        relative_path = out_path.relative_to(out_path.parents[2])
        save_path = relative_path.as_posix()

        return {"status": "ok", "order": payload, "save_path": save_path}