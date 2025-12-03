#!/usr/bin/env python3
"""
根据日期安装Python包的历史版本
用法: python install_packages_by_date.py <日期> <包列表>
日期格式: YYYYMMDD (如: 20250914)
"""

import sys
import re
import json
import subprocess
import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import urllib.request
import urllib.error


def parse_package_spec(package_spec: str) -> Tuple[str, Optional[str]]:
    """
    解析包规格，返回 (包名, 版本号)
    例如: "charset-normalizer==3.4.3" -> ("charset-normalizer", "3.4.3")
         "langchain_community" -> ("langchain_community", None)
    """
    # 匹配包名和版本号
    match = re.match(r'^([a-zA-Z0-9_\-\.]+)(?:==([0-9a-zA-Z\.]+))?$', package_spec.strip())
    if not match:
        raise ValueError(f"无法解析包规格: {package_spec}")
    
    package_name = match.group(1)
    version = match.group(2) if match.group(2) else None
    
    return package_name, version


def get_package_versions_with_dates(package_name: str) -> List[Tuple[str, datetime]]:
    """
    从PyPI获取包的所有版本及其发布日期
    返回: [(版本号, 发布日期), ...]
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            
        versions_with_dates = []
        releases = data.get('releases', {})
        
        for version, release_info in releases.items():
            if release_info:  # 确保有发布信息
                # 获取第一个文件的发布日期（通常是最早的）
                upload_time = release_info[0].get('upload_time')
                if upload_time:
                    try:
                        # PyPI返回的时间格式: "2024-01-15T10:30:00"
                        date_obj = datetime.fromisoformat(upload_time.replace('Z', '+00:00'))
                        versions_with_dates.append((version, date_obj))
                    except (ValueError, AttributeError):
                        continue
        
        return versions_with_dates
    
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"警告: 包 '{package_name}' 在PyPI上不存在")
        else:
            print(f"警告: 获取包 '{package_name}' 信息时出错: {e}")
        return []
    except Exception as e:
        print(f"警告: 获取包 '{package_name}' 信息时出错: {e}")
        return []


def find_latest_version_before_date(package_name: str, cutoff_date: datetime) -> Optional[str]:
    """
    找到指定日期之前的最新版本
    """
    versions_with_dates = get_package_versions_with_dates(package_name)
    
    if not versions_with_dates:
        return None
    
    # 过滤出在截止日期之前的版本，并按日期排序
    valid_versions = [
        (version, date) for version, date in versions_with_dates
        if date <= cutoff_date
    ]
    
    if not valid_versions:
        print(f"警告: 包 '{package_name}' 在 {cutoff_date.strftime('%Y-%m-%d')} 之前没有可用版本")
        return None
    
    # 按日期降序排序，取最新的
    valid_versions.sort(key=lambda x: x[1], reverse=True)
    
    return valid_versions[0][0]


def install_package(package_name: str, version: Optional[str] = None) -> bool:
    """
    使用pip安装指定版本的包
    """
    if version:
        package_spec = f"{package_name}=={version}"
    else:
        package_spec = package_name
    
    print(f"正在安装: {package_spec}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_spec],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ 成功安装: {package_spec}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 安装失败: {package_spec}")
        print(f"  错误信息: {e.stderr}")
        return False


def resolve_package_versions(date_str: str, packages: List[str], verbose: bool = True) -> Dict[str, Optional[str]]:
    """
    解析包列表，返回每个包对应的版本号
    
    Args:
        date_str: 日期字符串，格式 YYYYMMDD
        packages: 包列表，可以包含版本号（如 "package==1.0.0"）或不包含版本号
        verbose: 是否打印详细信息
    
    Returns:
        字典，键为包名，值为版本号（如果找不到则为None）
    """
    # 解析日期
    try:
        cutoff_date = datetime.strptime(date_str, "%Y%m%d")
        cutoff_date = cutoff_date.replace(hour=23, minute=59, second=59)
    except ValueError:
        raise ValueError(f"日期格式不正确，应为 YYYYMMDD，收到: {date_str}")
    
    if verbose:
        print(f"截止日期: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 分离有版本号的和没有版本号的包
    packages_with_version = {}
    packages_without_version = []
    
    for package_spec in packages:
        try:
            package_name, version = parse_package_spec(package_spec)
            if version:
                packages_with_version[package_name] = version
            else:
                packages_without_version.append(package_name)
        except ValueError as e:
            if verbose:
                print(f"错误: {e}")
            continue
    
    # 解析结果
    result = {}
    
    # 添加有指定版本的包
    result.update(packages_with_version)
    
    # 查找没有版本号的包
    if packages_without_version:
        if verbose:
            print("查询包版本信息...")
        
        for package_name in packages_without_version:
            if verbose:
                print(f"  查询: {package_name}")
            latest_version = find_latest_version_before_date(package_name, cutoff_date)
            result[package_name] = latest_version
            if verbose and latest_version:
                print(f"    -> {latest_version}")
            elif verbose:
                print(f"    -> 未找到合适版本")
    
    return result


def generate_requirements_txt(date_str: str, packages: List[str], output_file: str = "requirements.txt", verbose: bool = True) -> bool:
    """
    根据日期和包列表生成 requirements.txt 文件
    
    Args:
        date_str: 日期字符串，格式 YYYYMMDD
        packages: 包列表，可以包含版本号（如 "package==1.0.0"）或不包含版本号
        output_file: 输出的 requirements.txt 文件路径
        verbose: 是否打印详细信息
    
    Returns:
        是否成功生成文件
    """
    if verbose:
        print("=" * 60)
        print("生成 requirements.txt")
        print("=" * 60)
    
    # 解析包版本
    package_versions = resolve_package_versions(date_str, packages, verbose=verbose)
    
    # 生成 requirements.txt 内容
    requirements_lines = []
    failed_packages = []
    
    for package_name, version in sorted(package_versions.items()):
        if version:
            requirements_lines.append(f"{package_name}=={version}")
        else:
            failed_packages.append(package_name)
            if verbose:
                print(f"警告: 包 '{package_name}' 无法确定版本，将跳过")
    
    # 写入文件
    try:
        output_path = os.path.abspath(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(requirements_lines))
            f.write('\n')
        
        if verbose:
            print(f"\n✓ 成功生成: {output_path}")
            print(f"  共 {len(requirements_lines)} 个包")
            if failed_packages:
                print(f"  跳过 {len(failed_packages)} 个无法确定版本的包: {', '.join(failed_packages)}")
        
        return True
    except Exception as e:
        if verbose:
            print(f"✗ 生成文件失败: {e}")
        return False


def install_packages_by_date(date_str: str, packages: List[str], verbose: bool = True) -> None:
    """
    根据日期安装Python包的历史版本
    
    Args:
        date_str: 日期字符串，格式 YYYYMMDD
        packages: 包列表，可以包含版本号（如 "package==1.0.0"）或不包含版本号
        verbose: 是否打印详细信息
    """
    # 解析日期
    try:
        cutoff_date = datetime.strptime(date_str, "%Y%m%d")
        cutoff_date = cutoff_date.replace(hour=23, minute=59, second=59)
    except ValueError:
        raise ValueError(f"日期格式不正确，应为 YYYYMMDD，收到: {date_str}")
    
    if verbose:
        print(f"截止日期: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 解析包版本
    package_versions = resolve_package_versions(date_str, packages, verbose=False)
    
    # 分离有版本号的和没有版本号的包（用于显示）
    packages_with_version = []
    packages_without_version = []
    
    for package_spec in packages:
        try:
            package_name, version = parse_package_spec(package_spec)
            if version:
                packages_with_version.append((package_name, version))
            else:
                packages_without_version.append(package_name)
        except ValueError as e:
            if verbose:
                print(f"错误: {e}")
            continue
    
    # 安装有指定版本的包
    if packages_with_version:
        if verbose:
            print("=" * 60)
            print("安装有指定版本的包:")
            print("=" * 60)
        for package_name, version in packages_with_version:
            install_package(package_name, version)
    
    # 安装没有版本号的包（使用解析出的版本）
    if packages_without_version:
        if verbose:
            print("\n" + "=" * 60)
            print("安装指定日期之前的最新版本:")
            print("=" * 60)
        
        for package_name in packages_without_version:
            version = package_versions.get(package_name)
            if version:
                install_package(package_name, version)
            else:
                if verbose:
                    print(f"跳过包: {package_name} (无法找到合适的版本)")
    
    if verbose:
        print("\n" + "=" * 60)
        print("安装完成!")
        print("=" * 60)


def main():
    # 支持两种输入方式：
    # 1. 从标准输入读取（第一行是日期，第二行是包列表）
    # 2. 从命令行参数读取
    
    # 检查是否有 --generate-only 参数（只生成 requirements.txt，不安装）
    generate_only = "--generate-only" in sys.argv
    if generate_only:
        sys.argv.remove("--generate-only")
    
    if len(sys.argv) >= 3:
        # 方式1: 命令行参数
        date_str = sys.argv[1]
        packages = sys.argv[2:]
    elif not sys.stdin.isatty():
        # 方式2: 从标准输入读取
        lines = sys.stdin.read().strip().split('\n')
        if len(lines) < 2:
            print("错误: 需要两行输入，第一行是日期，第二行是包列表")
            sys.exit(1)
        date_str = lines[0].strip()
        packages = lines[1].strip().split()
    else:
        print("用法1: python install_packages_by_date.py <日期> <包1> <包2> ...")
        print("用法2: echo -e '日期\\n包列表' | python install_packages_by_date.py")
        print("用法3: python install_packages_by_date.py --generate-only <日期> <包1> <包2> ...")
        print("日期格式: YYYYMMDD (如: 20250914)")
        print("\n示例1:")
        print('  python install_packages_by_date.py 20250914 "langchain_community" "charset-normalizer==3.4.3"')
        print("\n示例2:")
        print('  echo -e "20250914\\nlangchain_community charset-normalizer==3.4.3" | python install_packages_by_date.py')
        print("\n示例3 (只生成 requirements.txt):")
        print('  python install_packages_by_date.py --generate-only 20250914 langchain_community charset-normalizer==3.4.3')
        sys.exit(1)
    
    if generate_only:
        # 只生成 requirements.txt
        generate_requirements_txt(date_str, packages)
    else:
        # 安装包
        install_packages_by_date(date_str, packages)


if __name__ == "__main__":
    main()

