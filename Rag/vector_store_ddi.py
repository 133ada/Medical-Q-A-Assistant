"""
DrugBank DDI Markdown → 本地向量库（增量构建版）
=============================================
只嵌入 Drug Interactions (DDI) 相关文件，来源目录：
  E:\\DrugBank_Details\\drugbank_md\\Drug Interactions

逻辑与原 vector_store.py 完全一致，仅修改：
  - MD_DIR         → 指向 DDI 专用目录
  - CHROMA_DIR     → 独立的 Chroma 持久化目录（与 other 分开存储）
  - HASH_FILE      → 独立的指纹文件

MAX_FILES = None → 处理所有待嵌入文件；设为整数可分批增量嵌入
"""

import os
import json
import hashlib
import time
from typing import Dict, List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from tqdm import tqdm
from pathlib import Path

# =============================================================================
# ★ 配置区
# =============================================================================

MD_DIR = r"E:\DrugBank_Details\drugbank_md\Drug Interactions"
CHROMA_DIR = r"E:\PycharmProjects\Knowledge Q&A Assistant\app\rag\chroma_drugbank_ddi"
HASH_FILE = r"E:\PycharmProjects\Knowledge Q&A Assistant\app\rag\chroma_drugbank_ddi\doc_hashes.json"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 80
ADD_CHUNK_BATCH = 50  # 每次 add_documents 的 chunk 上限




# =============================================================================


def _hash_file(filepath: Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_hashes() -> Dict[str, str]:
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_hashes(hashes: Dict[str, str]):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(hashes, f, ensure_ascii=False, indent=2)


def scan_md_files(md_dir: str) -> Dict[str, Path]:
    root = Path(md_dir)
    return {fp.stem: fp for fp in root.glob("*.md") if fp.is_file()}


def load_doc(source_key: str, filepath: Path) -> Document:
    text = filepath.read_text(encoding="utf-8", errors="replace")
    # source_key 形如 "DB00001_Lepirudin_DDI"
    parts = source_key.split("_", 1)
    return Document(
        page_content=text,
        metadata={
            "source": source_key,
            "drug_id": parts[0],
            "drug_name": parts[1] if len(parts) == 2 else "",
            "doc_type": "ddi",
            "filepath": str(filepath),
        },
    )


def add_chunks_in_batches(vectorstore: Chroma, chunks: List[Document]):
    for i in range(0, len(chunks), ADD_CHUNK_BATCH):
        vectorstore.add_documents(chunks[i: i + ADD_CHUNK_BATCH])


# ── 分批嵌入控制 ──────────────────────────────────────────────────────────────
# None  = 处理所有待嵌入文件
# 100   = 本次只处理前 100 个（按 DB 编号升序），下次继续追加

def build_vectorstore(embeddings, MAX_FILES: Optional[int] = 0) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    file_map = scan_md_files(MD_DIR)
    print(f"📁 [DDI] 扫描到 {len(file_map):,} 个 .md 文件")

    print("🔍 [DDI] 计算文件指纹...")
    new_hashes: Dict[str, str] = {}  # 它保存的是所有 md 文件的 hash 值
    for key, fp in tqdm(file_map.items(), desc="指纹计算", unit=" 文件", dynamic_ncols=True):
        new_hashes[key] = _hash_file(fp)

    old_hashes = _load_hashes()  # 保存的是已经嵌入好的文本内容的 hash 值
    current_keys = set(new_hashes)
    old_keys = set(old_hashes)
    # ✅ 只有 hash 变化的才需要重新嵌入
    added_keys = {k for k in current_keys if new_hashes[k] != old_hashes.get(k)}
    deleted_keys = old_keys - current_keys

    added_sorted = sorted(added_keys, key=lambda k: k.split("_")[0])
    total_pending = len(added_sorted)

    if MAX_FILES is not None and len(added_sorted) > MAX_FILES:
        this_batch = added_sorted[:MAX_FILES]
        remaining = total_pending - MAX_FILES
    else:
        this_batch = added_sorted
        remaining = 0

    print(f"\n📊 [DDI] 增量分析：")
    print(f"   待嵌入总量   : {total_pending:,} 个")
    print(f"   本次处理     : {len(this_batch):,} 个"
          + (f"  （{this_batch[0].split('_')[0]} → {this_batch[-1].split('_')[0]}）"
             if this_batch else ""))
    print(f"   本次跳过     : {remaining:,} 个（下次运行继续追加）")
    print(f"   ❌ 已删除    : {len(deleted_keys):,} 个")
    print(f"   ✅ 无变化    : {len(current_keys) - total_pending:,} 个\n")

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    existing_count = vectorstore._collection.count()
    print(f"📂 [DDI] {'加载已有' if existing_count > 0 else '初始化新'}向量库"
          f"（当前 {existing_count:,} 个向量块）")

    # ── 删除旧块（包含本批要重新嵌入的文件，避免重复）────────────────────────
    sources_to_remove = deleted_keys | set(this_batch)
    if sources_to_remove and existing_count > 0:
        print(f"\n🗑️  [DDI] 清理旧向量块...")
        existing = vectorstore.get(include=["metadatas"])
        ids_to_delete = [
            id_ for id_, meta in zip(existing["ids"], existing["metadatas"])
            if meta.get("source") in sources_to_remove
        ]
        if ids_to_delete:
            vectorstore.delete(ids=ids_to_delete)
            print(f"   已删除 {len(ids_to_delete):,} 个旧向量块")

    # ── 嵌入 ─────────────────────────────────────────────────────────────────
    failed_keys: List[str] = []

    if this_batch:
        total_chunks = 0
        t0 = time.time()
        print(f"\n➕ [DDI] 嵌入 {len(this_batch)} 个文件...\n")

        with tqdm(total=len(this_batch), desc="嵌入进度",
                  unit=" 文件", dynamic_ncols=True) as pbar:
            for key in this_batch:
                try:
                    doc = load_doc(key, file_map[key])
                    chunks = splitter.split_documents([doc])
                    if chunks:
                        add_chunks_in_batches(vectorstore, chunks)
                        total_chunks += len(chunks)
                    pbar.set_postfix_str(
                        f"{key.split('_')[0]}  chunks={len(chunks)}", refresh=False)
                except Exception as e:
                    failed_keys.append(key)
                    pbar.set_postfix_str(f"[错误]{key.split('_')[0]}: {e}", refresh=False)
                pbar.update(1)

        elapsed = time.time() - t0
        print(f"\n   ✅ 成功 {len(this_batch) - len(failed_keys)} / 失败 {len(failed_keys)}")
        print(f"   共写入 {total_chunks:,} 个向量块，耗时 {elapsed / 60:.1f} 分钟")
        if failed_keys:
            print(f"   ⚠️  失败列表: {failed_keys[:10]}")
    else:
        # ✅ 修复点：this_batch 为空时，什么都不做，不写入任何新指纹
        if not deleted_keys and not added_keys:
            print("✅ [DDI] 所有文件均无变化，直接使用缓存向量库")

    # 无论是否有 this_batch，都从旧 hashes 出发做增量更新
    merged_hashes = {**old_hashes}

    # 1. 移除已删除文件的指纹
    for k in deleted_keys:
        merged_hashes.pop(k, None)

    # 2. 只把本批成功的文件写入指纹（未处理的文件保持旧值或不存在）
    processed_ok = set(this_batch) - set(failed_keys)
    for k in processed_ok:
        merged_hashes[k] = new_hashes[k]

    # 下次运行时它们的 hash 仍不在 old_hashes 中 → 仍会出现在 added_keys → 继续处理

    _save_hashes(merged_hashes)

    if remaining > 0:
        print(f"\n💡 [DDI] 提示：还剩 {remaining:,} 个文件未嵌入，将 MAX_FILES 调大后再次运行。")

    final_count = vectorstore._collection.count()
    print(f"\n✅ [DDI] 向量库就绪，共 {final_count:,} 个向量块  |  目录: {CHROMA_DIR}\n")
    return vectorstore
