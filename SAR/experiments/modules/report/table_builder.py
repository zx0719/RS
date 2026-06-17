"""
table_builder.py — 从 objects[] 程序化生成附件表格

附件2：组成分布统计表（component_table）
附件3：装备分布统计表（equipment_table）

规则：
  - 所有数据严格来自 objects[]，禁止 LLM 自由生成。
  - 只包含 status != "FILTERED" 且 status != "REMOVED_BY_REVIEW" 的目标。
"""

from __future__ import annotations

from typing import Any

from modules.class_labels import get_class_name_cn, get_super_class_name_cn

# 审核状态中文映射
_REVIEW_STATUS_CN: dict[str, str] = {
    "UNREVIEWED": "未审核",
    "APPROVED": "已核实",
    "REJECTED": "已排除",
    "REVIEW_REQUIRED": "待审核",
}

# 需要过滤掉的 object status
_EXCLUDED_STATUSES = {"FILTERED", "REMOVED_BY_REVIEW"}


class TableBuilder:
    """从 objects[] 列表程序化构建附件表格。"""

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def build_component_table(self, objects: list[dict]) -> list[dict]:
        """构建附件2：组成分布统计表。

        字段：seq, target_type, sub_type, lon, lat

        Parameters
        ----------
        objects:
            Evidence Package 中的 objects[] 列表。

        Returns
        -------
        list[dict]
            每行一条目标记录。
        """
        rows: list[dict] = []
        seq = 1
        for obj in objects:
            if not self._is_valid(obj):
                continue

            cls = obj.get("class", {})
            geo = obj.get("geometry", {}).get("geo", {})

            super_class = cls.get("super_class", "")
            target_type = get_super_class_name_cn(super_class)
            sub_type = get_class_name_cn(
                cls.get("code"),
                cls.get("name_cn"),
                super_class,
            )

            lon = geo.get("center_lon")
            lat = geo.get("center_lat")

            rows.append(
                {
                    "seq": seq,
                    "target_type": target_type,
                    "sub_type": sub_type,
                    "lon": round(lon, 6) if lon is not None else None,
                    "lat": round(lat, 6) if lat is not None else None,
                }
            )
            seq += 1

        return rows

    def build_equipment_table(self, objects: list[dict]) -> list[dict]:
        """构建附件3：装备分布统计表。

        字段：seq, equipment_type, lon, lat, confidence, review_status

        Parameters
        ----------
        objects:
            Evidence Package 中的 objects[] 列表。

        Returns
        -------
        list[dict]
            每行一条目标记录。
        """
        rows: list[dict] = []
        seq = 1
        for obj in objects:
            if not self._is_valid(obj):
                continue

            cls = obj.get("class", {})
            score = obj.get("score", {})
            geo = obj.get("geometry", {}).get("geo", {})
            audit = obj.get("audit", {})

            equipment_type = get_class_name_cn(
                cls.get("code"),
                cls.get("name_cn"),
                cls.get("super_class"),
            )
            lon = geo.get("center_lon")
            lat = geo.get("center_lat")

            # 优先使用校准置信度
            confidence = score.get("calibrated_confidence") or score.get("confidence")
            if confidence is not None:
                confidence = round(float(confidence), 3)

            review_raw = audit.get("review_status", "UNREVIEWED")
            review_status_cn = _REVIEW_STATUS_CN.get(review_raw, review_raw)

            rows.append(
                {
                    "seq": seq,
                    "equipment_type": equipment_type,
                    "lon": round(lon, 6) if lon is not None else None,
                    "lat": round(lat, 6) if lat is not None else None,
                    "confidence": confidence,
                    "review_status": review_status_cn,
                }
            )
            seq += 1

        return rows

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid(obj: dict) -> bool:
        """判断目标是否应出现在表格中（排除已过滤/已删除的目标）。"""
        status = obj.get("status", "VALID")
        return status not in _EXCLUDED_STATUSES
