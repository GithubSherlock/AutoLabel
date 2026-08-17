"""数据集解压与路径管理 — 幂等、增量、筛选提取。

所有 ensure_* 函数已存在即跳过，不重复解压。
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from typing import Any

DATASETS_ROOT = Path.home() / "autodl-tmp" / "Documents" / "datasets"
ARCHIVE_ROOT = Path("/root/autodl-pub")


def extract_zip_members(
    zip_path: Path,
    dest: Path,
    member_prefix: str | None = None,
    max_files: int = 0,
) -> Path:
    """从 ZIP 中提取符合条件的成员。

    Args:
        zip_path: ZIP 文件路径。
        dest: 解压目标目录。
        member_prefix: 仅提取以此前缀开头的成员。
        max_files: 最多提取文件数（0 = 全部）。
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP 未找到: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if member_prefix:
            members = [m for m in members if m.startswith(member_prefix)]
        if max_files > 0:
            members = members[:max_files]

        # 检查是否已全部解压
        missing = [m for m in members if not (dest / m).exists()]
        if not missing:
            print(f"  ✓ 已存在 {len(members)} 个文件: {dest}")
            return dest

        print(f"  解压 {len(missing)} 个文件 → {dest} ...")
        for m in missing:
            zf.extract(m, dest)
    return dest


def extract_tar_members(
    tar_path: Path,
    dest: Path,
    member_prefix: str | None = None,
) -> Path:
    """从 tar.gz 中提取符合条件的成员。"""
    if not tar_path.exists():
        raise FileNotFoundError(f"tar 未找到: {tar_path}")

    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getnames()
        if member_prefix:
            members = [m for m in members if m.startswith(member_prefix)]

        missing = [m for m in members if not (dest / m).exists()]
        if not missing:
            print(f"  ✓ 已存在 {len(members)} 个文件: {dest}")
            return dest

        print(f"  解压 {len(missing)} 个文件 → {dest} ...")
        for m in missing:
            tf.extract(m, dest)
    return dest


def extract_zip_members_uniform(
    zip_path: Path,
    dest: Path,
    member_prefix: str | None = None,
    per_class: int = 0,
) -> Path:
    """从 ZIP 中按子目录分组、每组均匀提取前 per_class 个成员（确定性抽样）。

    适用于 ImageNet100 这类「wnid 目录即标签」的组织；组内排序后取前 N 张，
    保证可复现。现有 extract_zip_members 的 max_files 是全局截断（只覆盖前几个
    目录），无法做到每类均匀。

    Args:
        zip_path: ZIP 文件路径。
        dest: 解压目标目录。
        member_prefix: 仅提取以此前缀开头的成员。
        per_class: 每个子目录最多提取文件数（0 = 全部）。
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP 未找到: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]  # 过滤目录条目
        if member_prefix:
            members = [m for m in members if m.startswith(member_prefix)]

        # 按「前缀后第一段子目录」分组（imagenet100/n01558993/x.JPEG → n01558993）
        groups: dict[str, list[str]] = {}
        for m in members:
            rel = m[len(member_prefix):] if member_prefix else m
            parts = rel.split("/")
            if len(parts) < 2:
                continue
            groups.setdefault(parts[0], []).append(m)

        selected: list[str] = []
        for cls_members in groups.values():
            cls_members.sort()
            selected.extend(cls_members[:per_class] if per_class > 0 else cls_members)

        missing = [m for m in selected if not (dest / m).exists()]
        if not missing:
            print(f"  ✓ 已存在 {len(selected)} 个文件: {dest}")
            return dest

        print(f"  解压 {len(missing)} 个文件（{len(groups)} 个目录，每目录最多 {per_class} 个）→ {dest} ...")
        for m in missing:
            zf.extract(m, dest)
    return dest


# ================================================================
# 目标检测数据集
# ================================================================


def ensure_coco_val() -> Path:
    """解压 COCO2017 val（5,000 张图 + instances_val2017.json）。

    Returns:
        DATASETS_ROOT/COCO2017/
    """
    dest = DATASETS_ROOT / "COCO2017"
    dest.mkdir(parents=True, exist_ok=True)

    print("📦 解压 COCO2017 val ...")

    # 图像
    img_zip = ARCHIVE_ROOT / "COCO2017" / "val2017.zip"
    extract_zip_members(img_zip, dest)

    # 标注（仅 instances_val2017.json）
    anno_zip = ARCHIVE_ROOT / "COCO2017" / "annotations_trainval2017.zip"
    with zipfile.ZipFile(anno_zip, "r") as zf:
        anno_target = "annotations/instances_val2017.json"
        dest_file = dest / anno_target
        if not dest_file.exists():
            print(f"  解压 {anno_target} ...")
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            zf.extract(anno_target, dest)

    print(f"  ✓ COCO2017 val: {dest}")
    return dest


def ensure_voc2007() -> Path:
    """解压 VOC2007（test 4,952 张图 + XML 标注）。

    Returns:
        DATASETS_ROOT/VOCdevkit/VOC2007/
    """
    dest = DATASETS_ROOT / "VOCdevkit"
    dest.mkdir(parents=True, exist_ok=True)

    tar_path = ARCHIVE_ROOT / "VOCdevkit" / "VOC2007.tar.gz"

    # 检查是否已解压
    jpeg_dir = dest / "VOC2007" / "JPEGImages"
    anno_dir = dest / "VOC2007" / "Annotations"
    if jpeg_dir.exists() and anno_dir.exists():
        print(f"  ✓ 已存在: {dest / 'VOC2007'}")
        return dest

    print("📦 解压 VOC2007 (tar.gz) ...")
    extract_tar_members(tar_path, dest)
    print(f"  ✓ VOC2007: {dest / 'VOC2007'}")
    return dest


def ensure_kitti(max_images: int = 300) -> Path:
    """解压 KITTI object（前 max_images 张训练图 + 所有标注）。

    KITTI 图像 ZIP 12GB，解压全部较耗时。只提取前 max_images 张图。

    Returns:
        DATASETS_ROOT/KITTI/object/
    """
    dest = DATASETS_ROOT / "KITTI" / "object"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"📦 解压 KITTI object (最多 {max_images} 张图) ...")

    # 标注（5.4MB，先解压以获取文件列表）
    label_zip = ARCHIVE_ROOT / "KITTI" / "object" / "data_object_label_2.zip"
    extract_zip_members(label_zip, dest)

    # 从标注文件推断需要的图像名
    label_dir = dest / "training" / "label_2"
    if label_dir.exists():
        image_names = sorted([p.stem for p in label_dir.glob("*.txt")])[:max_images]
    else:
        image_names = []

    # 图像 — 筛选提取
    img_zip = ARCHIVE_ROOT / "KITTI" / "object" / "data_object_image_2.zip"
    target_members = [f"training/image_2/{name}.png" for name in image_names]

    img_dir = dest / "training" / "image_2"
    img_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(img_zip, "r") as zf:
        missing = [m for m in target_members if m in zf.namelist() and not (dest / m).exists()]
        if missing:
            print(f"  解压 {len(missing)} 张图像 → {img_dir} ...")
            for m in missing:
                zf.extract(m, dest)

    print(f"  ✓ KITTI object ({len(image_names)} 张图): {dest}")
    return dest


# ================================================================
# 实例分割数据集
# ================================================================


def ensure_cityscapes_val() -> Path:
    """解压 cityscapes val（504 张图 + gtFine 标注，instanceIds.png）。

    仅提取 val 分片（leftImg8bit/val + gtFine/val），不提取 train/test。

    Returns:
        DATASETS_ROOT/cityscapes/
    """
    dest = DATASETS_ROOT / "cityscapes"
    dest.mkdir(parents=True, exist_ok=True)

    print("📦 解压 cityscapes val ...")

    # gtFine（241MB，仅 val）
    gt_zip = ARCHIVE_ROOT / "cityscapes" / "gtFine_trainvaltest.zip"
    extract_zip_members(gt_zip, dest, member_prefix="gtFine/val")

    # 图像（11GB，仅 val）
    img_zip = ARCHIVE_ROOT / "cityscapes" / "leftImg8bit_trainvaltest.zip"
    extract_zip_members(img_zip, dest, member_prefix="leftImg8bit/val")

    print(f"  ✓ cityscapes val: {dest}")
    return dest


def ensure_nuimages_mini() -> Path:
    """解压 nuImages Mini（50 张图 + COCO JSON 标注，RLE mask）。

    nuImages devkit 要求 {dataroot}/v1.0-mini/ 子目录存在（含 table JSON）。
    解压后不 flatten — 保留 v1.0-mini/ 结构。

    Returns:
        DATASETS_ROOT/nuImages/
    """
    dest = DATASETS_ROOT / "nuImages"
    dest.mkdir(parents=True, exist_ok=True)

    # 检查是否已正确解压（必须包含 v1.0-mini/ 子目录）
    table_dir = dest / "v1.0-mini"
    samples_dir = dest / "samples"
    if table_dir.exists() and samples_dir.exists() and any(samples_dir.iterdir()):
        print(f"  ✓ 已存在: {dest}")
        return dest

    # 检查是否为旧的扁平结构（table JSON 在顶层），修复它
    table_files = list(dest.glob("*.json"))
    if table_files and samples_dir.exists():
        print(f"  修复目录结构: 将 {len(table_files)} 个 JSON 移入 v1.0-mini/ ...")
        table_dir.mkdir(parents=True, exist_ok=True)
        for f in table_files:
            f.rename(table_dir / f.name)
        print(f"  ✓ 已修复: {dest}")
        return dest

    tgz_path = ARCHIVE_ROOT / "nuScenes" / "nuImages" / "Mini" / "nuimages-v1.0-mini.tgz"

    if not tgz_path.exists():
        raise FileNotFoundError(f"nuImages Mini 未找到: {tgz_path}")

    print(f"📦 解压 nuImages Mini → {dest} ...")

    import subprocess
    subprocess.run(
        ["tar", "-xzf", str(tgz_path), "-C", str(dest)],
        check=True,
    )

    print(f"  ✓ nuImages Mini: {dest}")
    return dest


# ================================================================
# 航拍检测数据集
# ================================================================


def ensure_dota_val() -> Path:
    """解压 DOTA val（458 张航拍图 + HBB 水平框 + OBB 旋转框标注）。

    DOTA 是航拍图像旋转框目标检测数据集（15 类）。
    HBB（Task2 水平框）供通用检测 pipeline；OBB（Task1 旋转框，
    labels_obb/，文件前 2 行为 imagesource:/gsd: 头）供 OBB benchmark。
    仅提取 val 分片（train/test 不提取）。

    Returns:
        DATASETS_ROOT/DOTA/
    """
    dest = DATASETS_ROOT / "DOTA"
    dest.mkdir(parents=True, exist_ok=True)

    img_dir = dest / "images"
    label_dir = dest / "labels"
    obb_label_dir = dest / "labels_obb"

    # 快速幂等检查（三目录齐备才跳过）
    ready = (
        img_dir.exists() and label_dir.exists() and obb_label_dir.exists()
        and any(img_dir.iterdir()) and any(label_dir.iterdir())
        and any(obb_label_dir.iterdir())
    )
    if ready:
        print(f"  ✓ 已存在: {dest}")
        return dest

    print("📦 解压 DOTA val (航拍检测, ~3.2G) ...")

    # 图像 — val/images/part1.zip（458 张 PNG，~3.2GB）
    img_zip = ARCHIVE_ROOT / "DOTA" / "val" / "images" / "part1.zip"
    if not img_zip.exists():
        raise FileNotFoundError(f"DOTA val 图像未找到: {img_zip}")
    extract_zip_members(img_zip, dest, member_prefix="images/")

    # HBB 标注 — Val_Task2_gt.zip（水平框，无 header 行）
    # 解压后为 valset_reclabelTxt/Pxxxx.txt，展平到 labels/
    label_zip = ARCHIVE_ROOT / "DOTA" / "val" / "labelTxt-v1.0" / "Val_Task2_gt.zip"
    if not label_zip.exists():
        raise FileNotFoundError(f"DOTA val HBB 标注未找到: {label_zip}")
    with zipfile.ZipFile(label_zip, "r") as zf:
        txt_members = [m for m in zf.namelist() if m.endswith(".txt")]
        label_dir.mkdir(parents=True, exist_ok=True)
        missing = [m for m in txt_members if not (label_dir / Path(m).name).exists()]
        if missing:
            print(f"  解压 {len(missing)} 个 HBB 标注 → {label_dir} ...")
            for m in missing:
                # file inside zip: valset_reclabelTxt/P0003.txt
                data = zf.read(m)
                out_path = label_dir / Path(m).name
                out_path.write_bytes(data)

    # OBB 旋转框标注 — labelTxt.zip（Task1，前 2 行为 imagesource:/gsd: 头）
    # 解压后为 Pxxxx.txt，展平到 labels_obb/
    obb_zip = ARCHIVE_ROOT / "DOTA" / "val" / "labelTxt-v1.0" / "labelTxt.zip"
    if not obb_zip.exists():
        raise FileNotFoundError(f"DOTA val OBB 标注未找到: {obb_zip}")
    with zipfile.ZipFile(obb_zip, "r") as zf:
        txt_members = [m for m in zf.namelist() if m.endswith(".txt")]
        obb_label_dir.mkdir(parents=True, exist_ok=True)
        missing = [m for m in txt_members if not (obb_label_dir / Path(m).name).exists()]
        if missing:
            print(f"  解压 {len(missing)} 个 OBB 标注 → {obb_label_dir} ...")
            for m in missing:
                data = zf.read(m)
                out_path = obb_label_dir / Path(m).name
                out_path.write_bytes(data)

    print(f"  ✓ DOTA val: {dest}")
    return dest


# ================================================================
# 密集零售实例分割数据集
# ================================================================


def ensure_d2sa_val() -> Path:
    """解压 D2SA val（3,600 张零售货架图 + COCO JSON 标注）。

    D2SA 是密集零售货架商品实例分割数据集（60 SKU 类，RLE mask）。
    解压 annotations + 全部图像（~4.3G）。

    Returns:
        DATASETS_ROOT/D2SA/
    """
    dest = DATASETS_ROOT / "D2SA"
    dest.mkdir(parents=True, exist_ok=True)

    anno_dir = dest / "annotations"
    img_dir = dest / "images"

    # 快速幂等检查
    if anno_dir.exists() and img_dir.exists() and any(img_dir.iterdir()):
        print(f"  ✓ 已存在: {dest}")
        return dest

    print("📦 解压 D2SA (密集零售实例分割, ~4.3G) ...")

    # 标注（26MB tar.gz → annotations/*.json）
    anno_tar = ARCHIVE_ROOT / "D2SA" / "annotations.tar.gz"
    if not anno_tar.exists():
        raise FileNotFoundError(f"D2SA annotations 未找到: {anno_tar}")
    extract_tar_members(anno_tar, dest)

    # 图像（4.4GB tar.gz，内含 images/ 目录 → dest/images/）
    img_tar = ARCHIVE_ROOT / "D2SA" / "images.tar.gz"
    if not img_tar.exists():
        raise FileNotFoundError(f"D2SA images 未找到: {img_tar}")
    extract_tar_members(img_tar, dest)

    print(f"  ✓ D2SA: {dest}")
    return dest


# ================================================================
# 多目标跟踪 → 检测 Benchmark（仅行人）
# ================================================================


def ensure_mot17_frcnn() -> Path:
    """解析 MOT17 FRCNN 训练集路径（数据已解压）。

    MOT17 是行人多目标跟踪数据集，含 7 个场景 × 3 种检测器。
    仅使用 FRCNN 检测器的数据，共 7 个序列，5,316 张图。

    Returns:
        DATASETS_ROOT/MOT17/train/
    """
    dest = DATASETS_ROOT / "MOT17" / "train"
    if not dest.exists():
        raise FileNotFoundError(f"MOT17 FRCNN 数据未找到: {dest}")
    seqs = sorted(dest.glob("MOT17-*-FRCNN"))
    if not seqs:
        raise FileNotFoundError(f"未找到 MOT17 FRCNN 序列: {dest}")
    print(f"  ✓ MOT17 FRCNN: {len(seqs)} 个序列 ({sum(1 for s in seqs for _ in (s / 'img1').glob('*.jpg'))} 张图)")
    return dest


def ensure_mot20() -> Path:
    """解析 MOT20 训练集路径（数据已解压）。

    MOT20 是极密集行人跟踪数据集，4 个序列，8,931 张图。

    Returns:
        DATASETS_ROOT/MOT20/train/
    """
    dest = DATASETS_ROOT / "MOT20" / "train"
    if not dest.exists():
        raise FileNotFoundError(f"MOT20 数据未找到: {dest}")
    seqs = sorted(dest.glob("MOT20-*"))
    if not seqs:
        raise FileNotFoundError(f"未找到 MOT20 序列: {dest}")
    print(f"  ✓ MOT20: {len(seqs)} 个序列 ({sum(1 for s in seqs for _ in (s / 'img1').glob('*.jpg'))} 张图)")
    return dest


# ================================================================
# 图像分类数据集
# ================================================================


def ensure_imagenet100(per_class: int = 50) -> Path:
    """每类均匀抽样解压 ImageNet100（100 个 wnid 类，每类取 per_class 张）。

    归档 14GB / 126,689 图，全量解压在仅 CPU 服务器上既慢又占磁盘；
    抽样解压后评测子集 ≈ 100×per_class 张（50 → 5,000 张 ≈ 0.7GB）。

    Returns:
        DATASETS_ROOT/imagenet100/
    """
    dest = DATASETS_ROOT / "imagenet100"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"📦 解压 ImageNet100（每类 {per_class} 张）...")

    zip_path = ARCHIVE_ROOT / "ImageNet100" / "imagenet100.zip"
    extract_zip_members_uniform(zip_path, DATASETS_ROOT, member_prefix="imagenet100/", per_class=per_class)

    print(f"  ✓ ImageNet100: {dest}")
    return dest


def load_imagenet100_meta(
    devkit_archive: Path | None = None,
) -> dict[str, str]:
    """从 ILSVRC2012 devkit 的 meta.mat 解析 wnid → 英文类名（流式读取，不落盘）。

    meta.mat 含 1860 个 synset（字段 ILSVRC2012_ID / WNID / words）；
    words 形如 "kit fox, Vulpes macrotis"，取逗号前第一个词。已实测与
    torchvision ResNet18_Weights.DEFAULT.meta["categories"] 100/100 匹配。

    Args:
        devkit_archive: devkit tar.gz 路径，默认归档根下 t12 devkit。

    Returns:
        {wnid: 英文名} 映射。
    """
    from io import BytesIO

    if devkit_archive is None:
        devkit_archive = ARCHIVE_ROOT / "ImageNet" / "ILSVRC2012" / "ILSVRC2012_devkit_t12.tar.gz"
    if not devkit_archive.exists():
        raise FileNotFoundError(f"devkit 未找到: {devkit_archive}")

    import scipy.io as sio

    with tarfile.open(devkit_archive, "r:gz") as tf:
        member = next((m for m in tf.getnames() if m.endswith("meta.mat")), None)
        if member is None:
            raise FileNotFoundError(f"devkit 中未找到 meta.mat: {devkit_archive}")
        raw = tf.extractfile(member)
        if raw is None:
            raise FileNotFoundError(f"无法读取 {member}（空成员）")
        mat = sio.loadmat(BytesIO(raw.read()))

    synsets = mat["synsets"][:, 0]  # shape (1860, 1) struct 数组 → 按列取
    mapping: dict[str, str] = {}
    for s in synsets:
        wnid = str(s["WNID"][0])
        words = str(s["words"][0])
        mapping[wnid] = words.split(",")[0].strip()
    return mapping


def load_imagenet100_ground_truth(
    dataset_root: Path,
    max_images: int = 0,
) -> dict[str, dict[str, Any]]:
    """加载 ImageNet100 GT（wnid 目录名即标签 → 英文类名）。

    Args:
        dataset_root: ensure_imagenet100 返回的目录。
        max_images: 最多图像数（0=全部）；跨类均匀截断（每类按比例取），
            避免排序截断只覆盖少数类。

    Returns:
        {image_id(相对路径): {"file_name": str, "label": 英文名}}（单标签）
    """
    meta = load_imagenet100_meta()
    gt: dict[str, dict[str, Any]] = {}
    for cls_dir in sorted(dataset_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = meta.get(cls_dir.name, cls_dir.name)  # 未映射回退 wnid，不静默丢图
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in (".jpeg", ".jpg", ".png"):
                continue
            rel = str(img.relative_to(dataset_root))
            gt[rel] = {"file_name": rel, "label": label}

    if max_images > 0 and len(gt) > max_images:
        # 跨类均匀截断：每类按比例取前 N 张（确定性），再按全局排序截断
        by_cls: dict[str, list[str]] = {}
        for img_id, info in gt.items():
            by_cls.setdefault(str(info["label"]), []).append(img_id)
        per_cls = max(1, max_images // len(by_cls))
        picked: list[str] = []
        for ids in by_cls.values():
            picked.extend(ids[:per_cls])
        picked.sort()
        gt = {i: gt[i] for i in picked[:max_images]}
    return gt


# ================================================================
# 便捷函数：确保所有数据集
# ================================================================


def ensure_all() -> dict[str, Path]:
    """确保所有 10 个数据集已解压。

    Returns:
        {name: path} 映射。
    """
    return {
        "coco2017": ensure_coco_val(),
        "voc2007": ensure_voc2007(),
        "kitti": ensure_kitti(),
        "cityscapes": ensure_cityscapes_val(),
        "nuimages": ensure_nuimages_mini(),
        "dota": ensure_dota_val(),
        "d2sa": ensure_d2sa_val(),
        "mot17": ensure_mot17_frcnn(),
        "mot20": ensure_mot20(),
        "imagenet100": ensure_imagenet100(),
    }
