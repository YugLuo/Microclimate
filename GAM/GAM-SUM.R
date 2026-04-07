library(readxl) 
library(mgcv) 
library(eoffice) 
getwd()
data1<-read_excel('F:\\guan\\202512\\GAM-DATA.xlsx')
head(data1)

gamSUM<-gam(buffering~s(TCC)+s(LAI)+s(FAPAR)+s(CHT)+s(SLR)+s(TMP)+s(PCP)+s(ELEV)+s(SLP)+s(ASP),data = data1)
summary(gamSUM)
par(mfrow = c(4, 3))
plot(gamSUM,main = "SUM")

topptx(filename = "F:/guan/202512/GAM-R/GAM-SUM-V4.pptx",
       width = 8, height = 12)

smooth_vars <- lapply(gamAmerica$smooth, function(x) x$label) 
smooth_vars <- gsub("^s\\(|\\).*$", "", unlist(smooth_vars)) 

# 初始化数据框存储结果
gam_plot_data <- data.frame()

# 遍历每个变量，手动生成绘图数据
for (var in smooth_vars) {
  # 构造该变量的取值序列（覆盖数据范围）
  var_range <- range(data1[[var]], na.rm = TRUE)
  var_seq <- seq(var_range[1], var_range[2], length.out = 100) # 100个均匀取值点
  
  # 固定其他变量为均值，构造预测数据
  pred_data <- data1[1:100, ] # 复制数据结构
  pred_data[, smooth_vars] <- lapply(data1[, smooth_vars], mean, na.rm = TRUE) # 其他变量设为均值
  pred_data[[var]] <- var_seq # 当前变量取序列值
  
  # 预测拟合值和标准误
  pred_result <- predict(gamAmerica, newdata = pred_data, se.fit = TRUE)
  
  # 整理当前变量的绘图数据（全英文列名，避免语法错误）
  temp_data <- data.frame(
    variable = var,          # 变量名
    x_value = var_seq,       # 自变量取值
    fit_value = pred_result$fit, # 拟合值
    se_value = pred_result$se.fit, # 标准误
    ci_upper = pred_result$fit + 1.96 * pred_result$se.fit, # 95%置信区间上限
    ci_lower = pred_result$fit - 1.96 * pred_result$se.fit  # 95%置信区间下限
  )
  
  # 合并数据
  gam_plot_data <- rbind(gam_plot_data, temp_data)
}

# 导出到Excel
write_xlsx(gam_plot_data, path = "F:/guan/202512/GAM-R/GAM-sum_绘图数据.xlsx")