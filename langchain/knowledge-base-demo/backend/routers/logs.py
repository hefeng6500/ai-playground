from fastapi import APIRouter, HTTPException, Depends, Request, Query
from typing import List, Optional
from datetime import datetime, timedelta
from loguru import logger

from models import LogEntry, ApiResponse
from services.storage_service import StorageService

router = APIRouter()

def get_storage_service(request: Request) -> StorageService:
    """获取存储服务依赖"""
    return request.app.state.storage_service

@router.get("/", response_model=List[LogEntry])
async def get_logs(
    level: Optional[str] = Query(None, description="日志级别过滤 (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    start_time: Optional[datetime] = Query(None, description="开始时间过滤"),
    end_time: Optional[datetime] = Query(None, description="结束时间过滤"),
    limit: int = Query(100, ge=1, le=1000, description="返回条数限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    storage_service: StorageService = Depends(get_storage_service)
):
    """获取系统日志"""
    try:
        # 构建过滤条件
        filters = {}
        
        if level:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if level.upper() not in valid_levels:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效的日志级别: {level}. 有效值: {valid_levels}"
                )
            filters["level"] = level.upper()
        
        if start_time:
            filters["start_time"] = start_time
        
        if end_time:
            filters["end_time"] = end_time
        
        if search:
            filters["search"] = search
        
        # 获取日志
        logs = await storage_service.get_logs(
            filters=filters,
            limit=limit,
            offset=offset
        )
        
        logger.info(f"返回 {len(logs)} 条日志记录")
        return logs
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取日志失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取日志失败: {str(e)}"
        )

@router.get("/stats", response_model=dict)
async def get_log_stats(
    hours: int = Query(24, ge=1, le=168, description="统计时间范围（小时）"),
    storage_service: StorageService = Depends(get_storage_service)
):
    """获取日志统计信息"""
    try:
        # 计算时间范围
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # 获取指定时间范围内的日志
        logs = await storage_service.get_logs(
            filters={
                "start_time": start_time,
                "end_time": end_time
            },
            limit=10000  # 获取足够多的日志用于统计
        )
        
        # 统计各级别日志数量
        level_stats = {
            "DEBUG": 0,
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0
        }
        
        for log in logs:
            if log.level in level_stats:
                level_stats[log.level] += 1
        
        # 统计最近的错误
        recent_errors = [
            {
                "timestamp": log.timestamp,
                "level": log.level,
                "message": log.message[:100] + "..." if len(log.message) > 100 else log.message,
                "source": log.source
            }
            for log in logs 
            if log.level in ["ERROR", "CRITICAL"]
        ][:10]  # 最近10条错误
        
        stats = {
            "time_range": {
                "start_time": start_time,
                "end_time": end_time,
                "hours": hours
            },
            "total_logs": len(logs),
            "level_distribution": level_stats,
            "recent_errors": recent_errors,
            "error_rate": (
                (level_stats["ERROR"] + level_stats["CRITICAL"]) / len(logs) * 100
                if len(logs) > 0 else 0
            )
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"获取日志统计失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取日志统计失败: {str(e)}"
        )

@router.delete("/", response_model=ApiResponse)
async def clear_logs(
    before_date: Optional[datetime] = Query(None, description="删除此日期之前的日志"),
    level: Optional[str] = Query(None, description="只删除指定级别的日志"),
    confirm: bool = Query(False, description="确认删除操作"),
    storage_service: StorageService = Depends(get_storage_service)
):
    """清理系统日志"""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="请设置confirm=true来确认删除操作"
            )
        
        # 构建删除条件
        delete_filters = {}
        
        if before_date:
            delete_filters["before_date"] = before_date
        
        if level:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if level.upper() not in valid_levels:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效的日志级别: {level}. 有效值: {valid_levels}"
                )
            delete_filters["level"] = level.upper()
        
        # 如果没有指定任何条件，默认删除7天前的日志
        if not delete_filters:
            delete_filters["before_date"] = datetime.now() - timedelta(days=7)
        
        # 执行删除操作
        deleted_count = await storage_service.clear_logs(delete_filters)
        
        logger.info(f"清理日志完成，删除了 {deleted_count} 条记录")
        
        return ApiResponse(
            success=True,
            message=f"成功删除 {deleted_count} 条日志记录",
            data={
                "deleted_count": deleted_count,
                "filters": delete_filters
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清理日志失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"清理日志失败: {str(e)}"
        )

@router.get("/export", response_model=ApiResponse)
async def export_logs(
    format: str = Query("json", description="导出格式 (json, csv)"),
    level: Optional[str] = Query(None, description="日志级别过滤"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    storage_service: StorageService = Depends(get_storage_service)
):
    """导出日志数据"""
    try:
        if format not in ["json", "csv"]:
            raise HTTPException(
                status_code=400,
                detail="不支持的导出格式，支持: json, csv"
            )
        
        # 构建过滤条件
        filters = {}
        
        if level:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if level.upper() not in valid_levels:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效的日志级别: {level}. 有效值: {valid_levels}"
                )
            filters["level"] = level.upper()
        
        if start_time:
            filters["start_time"] = start_time
        
        if end_time:
            filters["end_time"] = end_time
        
        # 获取日志数据
        logs = await storage_service.get_logs(
            filters=filters,
            limit=10000  # 导出限制
        )
        
        # 生成导出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs_export_{timestamp}.{format}"
        
        # 准备导出数据
        if format == "json":
            export_data = [log.dict() for log in logs]
        else:  # csv
            # 简化的CSV格式
            export_data = [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "level": log.level,
                    "source": log.source,
                    "message": log.message.replace('\n', ' ').replace('\r', ' ')  # 清理换行符
                }
                for log in logs
            ]
        
        return ApiResponse(
            success=True,
            message=f"日志导出准备完成，共 {len(logs)} 条记录",
            data={
                "filename": filename,
                "format": format,
                "record_count": len(logs),
                "export_data": export_data
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出日志失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"导出日志失败: {str(e)}"
        )