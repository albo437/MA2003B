import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_curve, auc


def create_variable_boxplot(variable_index, df_list, figsize=(15, 8)):
    """
    Crea un boxplot para una variable específica de todas las estaciones
    
    Parameters:
    variable_index (int): Índice de la variable (1-15, ya que 0 es Date)
    figsize (tuple): Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    # Preparar datos para el boxplot
    data_for_boxplot = []
    station_labels = []
    variable_names = []

    # Obtener la variable especificada de cada estación
    for station_name, station_df in df_list.items():
        if variable_index < len(station_df.columns):
            variable_data = station_df.iloc[:, variable_index].dropna()
            data_for_boxplot.append(variable_data)
            station_labels.append(station_name)
            variable_names.append(station_df.columns[variable_index])
    
    # Crear el boxplot
    plt.boxplot(data_for_boxplot, tick_labels=station_labels)
    plt.xticks(rotation=45, ha='right')
    
    # Obtener el nombre común de la variable (sin el prefijo de estación)
    if variable_names:
        # Extraer la parte común del nombre de la variable
        common_var_name = variable_names[0].split(' ', 1)[1] if ' ' in variable_names[0] else variable_names[0]
        plt.title(f'Boxplot de {common_var_name} por Estación', fontsize=14, fontweight='bold')
        
        # Determinar la unidad de medida para el label del eje Y
        if 'ppm' in common_var_name:
            ylabel = 'Concentración (ppm)'
        elif 'ppb' in common_var_name:
            ylabel = 'Concentración (ppb)'
        elif 'ug/m3' in common_var_name:
            ylabel = 'Concentración (μg/m³)'
        elif 'mmhg' in common_var_name:
            ylabel = 'Presión (mmHg)'
        elif 'mm/hr' in common_var_name:
            ylabel = 'Precipitación (mm/hr)'
        elif '%' in common_var_name:
            ylabel = 'Porcentaje (%)'
        elif 'KW/m2' in common_var_name:
            ylabel = 'Radiación Solar (KW/m²)'
        elif 'degC' in common_var_name:
            ylabel = 'Temperatura (°C)'
        elif 'KMPH' in common_var_name:
            ylabel = 'Velocidad del Viento (km/h)'
        elif 'DEG' in common_var_name:
            ylabel = 'Dirección del Viento (grados)'
        else:
            ylabel = 'Valor'
            
        plt.ylabel(ylabel)
    
    plt.xlabel('Estaciones')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Imprimir estadísticas básicas
    print(f"Variable {variable_index}: {common_var_name if variable_names else 'Desconocida'}")
    print("-" * 60)
    for i, (station_name, station_df) in enumerate(df_list.items()):
        if variable_index < len(station_df.columns):
            variable_data = station_df.iloc[:, variable_index].dropna()
            print(f"{station_name}: Media={variable_data.mean():.2f}, Mediana={variable_data.median():.2f}, "
                  f"Desv.Est={variable_data.std():.2f}")

def create_correlation_heatmap(variable_index, df_list,figsize=(12, 10)):
    """
    Crea un mapa de correlación (heatmap) para una variable específica entre todas las estaciones
    
    Parameters:
    variable_index (int): Índice de la variable (1-15, ya que 0 es Date)
    figsize (tuple): Tamaño de la figura
    """
    
    # Crear un DataFrame con todas las estaciones para la variable específica
    correlation_data = pd.DataFrame()
    variable_name = None

    for station_name, station_df in df_list.items():
        if variable_index < len(station_df.columns):
            variable_data = station_df.iloc[:, variable_index]
            correlation_data[station_name] = variable_data
            if variable_name is None:
                variable_name = station_df.columns[variable_index]
    
    # Calcular la matriz de correlación
    correlation_matrix = correlation_data.corr()
    
    # Crear el heatmap
    plt.figure(figsize=figsize)
    
    # Usar seaborn para crear un heatmap más atractivo
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)  # Máscara para mostrar solo la mitad inferior
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                fmt='.2f',
                square=True,
                annot_kws={'size': 8},
                cbar_kws={"shrink": .8})
    
    # Obtener el nombre común de la variable
    common_var_name = variable_name.split(' ', 1)[1] if ' ' in variable_name else variable_name
    
    plt.title(f'Matriz de Correlación - {common_var_name}\nCorrelaciones entre Estaciones', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Imprimir estadísticas de correlación
    print(f"Variable {variable_index}: {common_var_name}")
    print("-" * 60)
    
    # Encontrar las correlaciones más altas y más bajas (excluyendo la diagonal)
    correlation_values = correlation_matrix.values.copy()
    np.fill_diagonal(correlation_values, np.nan)  # Eliminar la diagonal
    
    # Obtener índices de correlación máxima y mínima
    max_corr_idx = np.unravel_index(np.nanargmax(correlation_values), correlation_values.shape)
    min_corr_idx = np.unravel_index(np.nanargmin(correlation_values), correlation_values.shape)
    
    max_corr_value = correlation_values[max_corr_idx]
    min_corr_value = correlation_values[min_corr_idx]
    
    stations_list = list(correlation_matrix.columns)
    
    print(f"Correlación más alta: {stations_list[max_corr_idx[0]]} - {stations_list[max_corr_idx[1]]}: {max_corr_value:.3f}")
    print(f"Correlación más baja: {stations_list[min_corr_idx[0]]} - {stations_list[min_corr_idx[1]]}: {min_corr_value:.3f}")
    
    # Estadísticas generales de correlación
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    correlations = upper_triangle.stack().values
    
    print(f"Correlación promedio entre estaciones: {np.mean(correlations):.3f}")
    print(f"Desviación estándar de correlaciones: {np.std(correlations):.3f}")
    print(f"Correlaciones > 0.8: {np.sum(correlations > 0.8)} de {len(correlations)} pares")
    print(f"Correlaciones < 0.5: {np.sum(correlations < 0.5)} de {len(correlations)} pares")

def analyze_station_correlations(df_station, station_name, min_correlation=0.3, figsize=(12, 10)):
    """
    Analiza las correlaciones entre todas las variables de una estación específica.
    
    Parámetros:
    -----------
    df_station : pd.DataFrame
        DataFrame de una estación con todas sus variables.
    station_name : str
        Nombre de la estación para títulos y reportes.
    min_correlation : float
        Umbral mínimo de correlación para considerar significativa.
    figsize : tuple
        Tamaño de la figura del heatmap.
    
    Retorna:
    --------
    dict : Diccionario con matrices de correlación y análisis estadístico.
    """
    
    print(f"🔗" * 20)
    print(f"ANÁLISIS DE CORRELACIONES INTRA-ESTACIÓN: {station_name}")
    print(f"🔗" * 20)
    
    # Obtener solo las columnas numéricas (excluir Date)
    numeric_cols = df_station.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print(f"❌ No hay suficientes variables numéricas para análisis de correlación")
        return {}
    
    print(f"📊 Variables analizadas: {len(numeric_cols)}")
    print(f"📈 Datos disponibles: {len(df_station)} registros")
    
    # Calcular matriz de correlación
    correlation_matrix = df_station[numeric_cols].corr()
    
    # Crear el heatmap
    plt.figure(figsize=figsize)
    
    # Máscara para el triángulo superior (opcional, para no duplicar información)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Crear heatmap con seaborn
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                mask=mask,
                cbar_kws={"shrink": .8},
                fmt='.2f',
                linewidths=0.5)
    
    plt.title(f'Matriz de Correlaciones - Estación {station_name}\n'
              f'({len(numeric_cols)} variables, {len(df_station)} observaciones)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Análisis de correlaciones significativas
    print(f"\n📋 ANÁLISIS DE CORRELACIONES SIGNIFICATIVAS (|r| ≥ {min_correlation}):")
    print("=" * 80)
    
    significant_correlations = []
    
    # Buscar correlaciones significativas
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            var1 = correlation_matrix.columns[i]
            var2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            
            if abs(corr_value) >= min_correlation:
                # Limpiar nombres de variables (quitar prefijo de estación)
                clean_var1 = var1.split(' ', 1)[1] if ' ' in var1 else var1
                clean_var2 = var2.split(' ', 1)[1] if ' ' in var2 else var2
                
                significant_correlations.append({
                    'Variable_1': clean_var1,
                    'Variable_2': clean_var2,
                    'Correlacion': round(corr_value, 3),
                    'Fuerza': categorize_correlation(abs(corr_value)),
                    'Direccion': 'Positiva' if corr_value > 0 else 'Negativa'
                })
    
    if significant_correlations:
        # Ordenar por valor absoluto de correlación
        significant_df = pd.DataFrame(significant_correlations)
        significant_df = significant_df.sort_values('Correlacion', key=abs, ascending=False)
        
        print(f"Se encontraron {len(significant_correlations)} correlaciones significativas:")
        print(significant_df.to_string(index=False))
        
        # Estadísticas por categoría
        print(f"\n📊 DISTRIBUCIÓN POR FUERZA DE CORRELACIÓN:")
        strength_counts = significant_df['Fuerza'].value_counts()
        for strength, count in strength_counts.items():
            print(f"  • {strength}: {count} correlaciones")
        
        # Top correlaciones
        print(f"\n🏆 TOP 5 CORRELACIONES MÁS FUERTES:")
        print("-" * 50)
        for _, row in significant_df.head(5).iterrows():
            print(f"  {row['Variable_1']} ↔ {row['Variable_2']}: "
                  f"r = {row['Correlacion']:.3f} ({row['Fuerza']}, {row['Direccion']})")
    
    else:
        print(f"❌ No se encontraron correlaciones significativas con |r| ≥ {min_correlation}")
        print(f"💡 Considera reducir el umbral (ej. 0.2) para detectar correlaciones más débiles")
    
    # Estadísticas generales
    print(f"\n📈 ESTADÍSTICAS GENERALES:")
    print("-" * 40)
    print(f"  • Correlación promedio (valor absoluto): {correlation_matrix.abs().mean().mean():.3f}")
    print(f"  • Correlación máxima: {correlation_matrix.abs().max().max():.3f}")
    print(f"  • Variables más correlacionadas con otras:")
    
    # Variables con mayor correlación promedio
    avg_correlations = correlation_matrix.abs().mean().sort_values(ascending=False)
    for i, (var, avg_corr) in enumerate(avg_correlations.head(3).items()):
        clean_var = var.split(' ', 1)[1] if ' ' in var else var
        print(f"    {i+1}. {clean_var}: {avg_corr:.3f}")
    
    return {
        'correlation_matrix': correlation_matrix,
        'significant_correlations': significant_correlations if significant_correlations else [],
        'station_name': station_name,
        'n_variables': len(numeric_cols),
        'n_observations': len(df_station)
    }

def categorize_correlation(abs_corr):
    """Categoriza la fuerza de una correlación basada en su valor absoluto."""
    if abs_corr >= 0.8:
        return "Muy Fuerte"
    elif abs_corr >= 0.6:
        return "Fuerte"
    elif abs_corr >= 0.4:
        return "Moderada"
    elif abs_corr >= 0.2:
        return "Débil"
    else:
        return "Muy Débil"

def correct_outliers(dfs_dict, method='rolling_zscore', window_size=24, z_threshold=5, verbose=True):
    """
    Corrige outliers en series temporales de contaminantes y variables meteorológicas.
    - Detecta outliers físicos y estadísticos.
    - Sustituye con interpolación si son pocos consecutivos.
    - Si hay >3 consecutivos, usa KNNImputer.
    - Devuelve resumen con detalle físico/estadístico.
    """
    
    PHYSICAL_LIMITS = {
        'temp':         {'min': -50,  'max': 60,   'name': 'Temperatura (°C)'},
        'humidity':     {'min': 0,    'max': 200,  'name': 'Humedad Relativa (%)'},
        'radiation':    {'min': 0,    'max': 4.0,  'name': 'Radiación Solar (kW/m²)'},
        'precipitation':{'min': 0,    'max': 300,  'name': 'Precipitación (mm/h)'},
        'pressure':     {'min': 600,  'max': 820,  'name': 'Presión Atmosférica (mm Hg)'},
        'wind_speed':   {'min': 0,    'max': 250,  'name': 'Velocidad del Viento (km/h)'},
        'wind_dir':     {'min': 0,    'max': 360,  'name': 'Dirección del Viento (°)'},
        'pm10':         {'min': 0, 'max': 10000, 'name': 'PM10 (µg/m³)'},
        'pm25':         {'min': 0, 'max': 5000,  'name': 'PM2.5 (µg/m³)'},
        'o3':           {'min': 0, 'max': 5000,  'name': 'Ozono (ppb)'},
        'so2':          {'min': 0, 'max': 5000,  'name': 'Dióxido de Azufre (ppb)'},
        'no2':          {'min': 0, 'max': 50000, 'name': 'Dióxido de Nitrógeno (ppb)'},
        'no':           {'min': 0, 'max': 50000, 'name': 'Monóxido de Nitrógeno (ppb)'},
        'nox':          {'min': 0, 'max': 100000,'name': 'Óxidos de Nitrógeno (ppb)'},
        'co':           {'min': 0, 'max': 200,   'name': 'Monóxido de Carbono (ppm)'}, 
    }

    COLUMN_INDEX_TO_TYPE = {
        1: 'co', 2: 'no', 3: 'no2', 4: 'nox', 5: 'o3', 6: 'pm10', 7: 'pm25', 8: 'pressure',
        9: 'precipitation', 10: 'humidity', 11: 'so2', 12: 'radiation', 13: 'temp',
        14: 'wind_speed', 15: 'wind_dir'
    }

    def rolling_zscore_outliers(data, window=window_size, threshold=z_threshold):
        if len(data) < window: window = max(3, len(data) // 2)
        rolling_mean = data.rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = data.rolling(window=window, center=True, min_periods=1).std()
        z_scores = np.abs((data - rolling_mean) / rolling_std)
        return z_scores > threshold

    def apply_physical_limits(data, var_type):
        if var_type not in PHYSICAL_LIMITS:
            return pd.Series(False, index=data.index)
        limits = PHYSICAL_LIMITS[var_type]
        return (data < limits['min']) | (data > limits['max'])

    corrected_dfs = {}
    correction_summary = {}
    
    if verbose:
        print("🔧 CORRECCIÓN DE OUTLIERS (Interpolación + KNN para >3)")
        print(f"Método: {method} | Ventana: {window_size} | Umbral Z: {z_threshold}\n" + "="*60)
    
    total_corrections = 0
    
    for station_name, df in dfs_dict.items():
        if verbose:
            print(f"\n🏭 Procesando estación: {station_name}")
        df_corrected = df.copy()
        station_corrections = {}
        station_total_corrections = 0
        
        for col_idx in range(1, len(df.columns)):
            col_name = df.columns[col_idx]
            var_type = COLUMN_INDEX_TO_TYPE.get(col_idx)
            
            if not var_type:
                continue

            original_series = df.iloc[:, col_idx].copy()
            working_series = original_series.copy()
            
            if working_series.isna().all():
                continue
            
            # --- 1. Outliers físicos ---
            physical_outliers = apply_physical_limits(working_series, var_type)
            num_physical = int(physical_outliers.sum())
            working_series[physical_outliers] = np.nan
            
            # --- 2. Outliers estadísticos ---
            statistical_outliers = rolling_zscore_outliers(working_series, window_size, z_threshold)
            num_statistical = int(statistical_outliers.sum())
            working_series[statistical_outliers] = np.nan

            # --- 3. Interpolación inicial ---
            interpolated = working_series.interpolate(method='linear', limit_direction='both')

            # --- 4. KNN en bloques largos ---
            na_mask = interpolated.isna()
            used_knn = False
            if na_mask.any():
                runs = (na_mask != na_mask.shift()).cumsum()
                run_lengths = na_mask.groupby(runs).transform('sum')
                long_blocks = (na_mask) & (run_lengths > 3)

                if long_blocks.any():
                    used_knn = True
                    imputer = KNNImputer(n_neighbors=3)
                    arr = interpolated.values.reshape(-1, 1)
                    arr_imputed = imputer.fit_transform(arr)
                    interpolated = pd.Series(arr_imputed.ravel(), index=interpolated.index)

            # --- 5. Guardar resultado ---
            df_corrected.iloc[:, col_idx] = interpolated
            final_outliers_mask = working_series.isna()
            outlier_count = int(final_outliers_mask.sum())
            
            if outlier_count > 0:
                station_total_corrections += outlier_count
                total_corrections += outlier_count

            station_corrections[col_name] = {
                'outlier_count': outlier_count,
                'outlier_percentage': outlier_count / len(original_series) * 100 if len(original_series) > 0 else 0,
                'physical_outliers': num_physical,
                'statistical_outliers': num_statistical,
                'original_min': working_series.min(),
                'original_max': working_series.max(),
                'corrected_min': interpolated.min(),
                'corrected_max': interpolated.max(),
                'variable_type': var_type,
                'method_used': 'KNN' if used_knn else 'Interpolation'
            }
        
        if verbose and station_total_corrections > 0:
            print(f"  ✅ {station_total_corrections} outliers corregidos en total")
        elif verbose:
            print(f"  ✨ Sin outliers detectados")
        
        corrected_dfs[station_name] = df_corrected
        correction_summary[station_name] = station_corrections
    
    if verbose:
        print(f"\n🎯 CORRECCIÓN COMPLETADA: {total_corrections} outliers corregidos en total")
    
    return corrected_dfs, correction_summary

def generate_correction_report(correction_summary, dfs_original, dfs_corrected):
    """
    Genera un reporte detallado de las correcciones realizadas
    """
    print("\n" + "="*80)
    print("📋 REPORTE DETALLADO DE CORRECCIONES")
    print("="*80)
    
    total_corrections = 0
    stations_with_corrections = 0
    
    for station_name, corrections in correction_summary.items():
        if corrections:
            station_total = sum(corr.get('outlier_count', 0) for corr in corrections.values())
            if station_total > 0:
                stations_with_corrections += 1
                total_corrections += station_total
                
                print(f"\n📍 {station_name}: {station_total} outliers corregidos")
                
                # Mostrar solo las variables con correcciones significativas
                for var_name, corr_info in corrections.items():
                    outlier_count = corr_info.get('outlier_count', 0)
                    if outlier_count > 0:
                        physical = corr_info.get('physical_outliers', 0)
                        statistical = corr_info.get('statistical_outliers', 0)
                        print(f"   • {var_name}: {outlier_count} outliers ({physical}F, {statistical}E)")
    
    if stations_with_corrections == 0:
        print("\n✨ No se detectaron outliers en ninguna estación")
    
    print(f"\n" + "="*80)
    print("📊 RESUMEN GENERAL:")
    print(f"• Estaciones procesadas: {len(correction_summary)}")
    print(f"• Estaciones con correcciones: {stations_with_corrections}")
    print(f"• Total de outliers corregidos: {total_corrections}")
    print("="*80)

def validate_corrections(dfs_original, dfs_corrected):
    """
    Valida que las correcciones se realizaron correctamente
    """
    print("\n🔍 VALIDACIÓN DE CORRECCIONES:")
    print("-" * 50)
    
    stations_with_changes = 0
    total_extreme_remaining = 0
    
    for station_name in dfs_original.keys():
        df_orig = dfs_original[station_name]
        df_corr = dfs_corrected[station_name]
        
        # Verificar cambios significativos
        changes_detected = False
        extreme_values_remaining = 0
        
        for col_idx in range(1, len(df_orig.columns)):
            orig_data = df_orig.iloc[:, col_idx]
            corr_data = df_corr.iloc[:, col_idx]
            
            # Contar cambios
            changes = (orig_data != corr_data).sum()
            if changes > 0:
                changes_detected = True
            
            # Verificar valores extremos restantes (muy básico)
            if not corr_data.empty and pd.api.types.is_numeric_dtype(corr_data):
                q99 = corr_data.quantile(0.99)
                q01 = corr_data.quantile(0.01)
                if pd.notna(q99) and pd.notna(q01):
                    extreme_remaining = ((corr_data > q99 * 3) | (corr_data < q01 * 3)).sum()
                    extreme_values_remaining += extreme_remaining
        
        if changes_detected:
            stations_with_changes += 1
        
        total_extreme_remaining += extreme_values_remaining
        
        status = "✅ Procesado" if changes_detected else "ℹ️  Sin cambios"
        print(f"{status} {station_name}")
    
    print("-" * 50)
    print(f"📊 Estaciones con cambios: {stations_with_changes}/{len(dfs_original)}")
    print(f"⚠️  Valores extremos restantes: {total_extreme_remaining}")
    print("-" * 50)

def compare_datasets_statistics(dfs_original, dfs_corrected, variable_index):
    """
    Compara estadísticas entre datasets originales y corregidos para una variable específica
    """
    print(f"\n📊 COMPARACIÓN ESTADÍSTICA - Variable {variable_index}")
    print("="*70)
    
    comparison_stats = []
    
    for station_name in dfs_original.keys():
        df_orig = dfs_original[station_name]
        df_corr = dfs_corrected[station_name]
        
        if variable_index < len(df_orig.columns):
            var_name = df_orig.columns[variable_index]
            
            orig_data = df_orig.iloc[:, variable_index].dropna()
            corr_data = df_corr.iloc[:, variable_index].dropna()
            
            # Calcular estadísticas
            stats_comparison = {
                'Station': station_name,
                'Variable': var_name,
                'Original_Mean': orig_data.mean(),
                'Corrected_Mean': corr_data.mean(),
                'Original_Std': orig_data.std(),
                'Corrected_Std': corr_data.std(),
                'Original_Min': orig_data.min(),
                'Corrected_Min': corr_data.min(),
                'Original_Negatives': (orig_data < 0).sum(),
                'Corrected_Negatives': (corr_data < 0).sum()
            }
            
            comparison_stats.append(stats_comparison)
            
            # Mostrar comparación por estación
            print(f"\n🏭 {station_name}:")
            print(f"   Media: {orig_data.mean():.3f} → {corr_data.mean():.3f}")
            print(f"   Mínimo: {orig_data.min():.3f} → {corr_data.min():.3f}")
            print(f"   Negativos: {(orig_data < 0).sum()} → {(corr_data < 0).sum()}")
    
    return pd.DataFrame(comparison_stats)

def create_before_after_plots(dfs_original, dfs_corrected, variable_index, station_name):
    """
    Crea gráficos de comparación antes y después para una variable y estación específica
    """
    df_orig = dfs_original[station_name]
    df_corr = dfs_corrected[station_name]
    
    if variable_index < len(df_orig.columns):
        var_name = df_orig.columns[variable_index]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        orig_data = df_orig.iloc[:, variable_index]
        corr_data = df_corr.iloc[:, variable_index]
        
        # Serie temporal - Original
        axes[0, 0].plot(orig_data, alpha=0.7, color='red', label='Original')
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 0].set_title(f'Serie Temporal Original - {station_name}')
        axes[0, 0].set_ylabel('Valor')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Serie temporal - Corregido
        axes[0, 1].plot(corr_data, alpha=0.7, color='green', label='Corregido')
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_title(f'Serie Temporal Corregida - {station_name}')
        axes[0, 1].set_ylabel('Valor')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histograma comparativo
        axes[1, 0].hist(orig_data.dropna(), bins=30, alpha=0.6, color='red', label='Original')
        axes[1, 0].hist(corr_data.dropna(), bins=30, alpha=0.6, color='green', label='Corregido')
        axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Distribución Comparativa')
        axes[1, 0].set_xlabel('Valor')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Boxplot comparativo
        axes[1, 1].boxplot([orig_data.dropna(), corr_data.dropna()], 
                          labels=['Original', 'Corregido'])
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Boxplot Comparativo')
        axes[1, 1].set_ylabel('Valor')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Análisis Antes/Después - Variable {variable_index}: {var_name}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Estadísticas detalladas
        print(f"\n📈 ESTADÍSTICAS DETALLADAS:")
        print(f"Variable: {var_name}")
        print(f"Estación: {station_name}")
        print("-" * 50)
        print("ORIGINAL:")
        print(f"  Media: {orig_data.mean():.4f}")
        print(f"  Mediana: {orig_data.median():.4f}")
        print(f"  Desv. Est.: {orig_data.std():.4f}")
        print(f"  Min: {orig_data.min():.4f}")
        print(f"  Max: {orig_data.max():.4f}")
        print(f"  Valores negativos: {(orig_data < 0).sum()}")
        print(f"  Valores faltantes: {orig_data.isna().sum()}")
        
        print("\nCORREGIDO:")
        print(f"  Media: {corr_data.mean():.4f}")
        print(f"  Mediana: {corr_data.median():.4f}")
        print(f"  Desv. Est.: {corr_data.std():.4f}")
        print(f"  Min: {corr_data.min():.4f}")
        print(f"  Max: {corr_data.max():.4f}")
        print(f"  Valores negativos: {(corr_data < 0).sum()}")
        print(f"  Valores faltantes: {corr_data.isna().sum()}")

def create_compact_histogram(variable_index, df_list, figsize=(16, 12)):
    """
    Crea histogramas compactos para una variable específica de todas las estaciones
    con pruebas de normalidad
    
    Parameters:
    variable_index (int): Índice de la variable (1-15, ya que 0 es Date)
    figsize (tuple): Tamaño de la figura
    """
    
    # Preparar datos
    data_by_station = {}
    variable_name = None

    for station_name, station_df in df_list.items():
        if variable_index < len(station_df.columns):
            variable_data = station_df.iloc[:, variable_index].dropna()
            data_by_station[station_name] = variable_data
            if variable_name is None:
                variable_name = station_df.columns[variable_index]
    
    # Obtener el nombre común de la variable
    common_var_name = variable_name.split(' ', 1)[1] if ' ' in variable_name else variable_name
    
    # Crear subplot con 3 filas y 5 columnas para 15 estaciones
    fig, axes = plt.subplots(3, 5, figsize=figsize)
    axes = axes.flatten()
    
    # Lista para almacenar resultados de normalidad
    normality_results = []
    
    for i, (station_name, data) in enumerate(data_by_station.items()):
        ax = axes[i]
        
        # Crear histograma
        n, bins, patches = ax.hist(data, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Añadir curva normal teórica
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal teórica')
        
        # Prueba de normalidad (Shapiro-Wilk para muestras pequeñas, Anderson-Darling para grandes)
        if len(data) <= 5000:
            stat, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
        else:
            # Para muestras muy grandes, usar Kolmogorov-Smirnov
            stat, p_value = stats.kstest(data, 'norm', args=(mu, sigma))
            test_name = "Kolmogorov-Smirnov"
        
        # Determinar si es normal (p > 0.05)
        is_normal = p_value > 0.05
        color = 'green' if is_normal else 'red'
        
        # Título con información de normalidad
        ax.set_title(f'{station_name}\np={p_value:.4f}', fontsize=10, color=color, fontweight='bold')
        ax.set_xlabel('Valor', fontsize=8)
        ax.set_ylabel('Densidad', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        
        # Guardar resultado
        normality_results.append({
            'Station': station_name,
            'Test': test_name,
            'Statistic': stat,
            'P-value': p_value,
            'Is_Normal': is_normal,
            'Mean': mu,
            'Std': sigma,
            'Skewness': stats.skew(data),
            'Kurtosis': stats.kurtosis(data)
        })
    
    # Título general
    plt.suptitle(f'Histogramas de {common_var_name} por Estación\n(Verde: Normal, Rojo: No Normal)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    # Crear resumen estadístico
    results_df = pd.DataFrame(normality_results)
    
    print(f"Variable {variable_index}: {common_var_name}")
    print("=" * 80)
    print(f"Estaciones con distribución NORMAL (p > 0.05): {results_df['Is_Normal'].sum()} de {len(results_df)}")
    print(f"Estaciones con distribución NO NORMAL (p ≤ 0.05): {(~results_df['Is_Normal']).sum()} de {len(results_df)}")
    
    print(f"\nP-valores promedio por prueba:")
    for test in results_df['Test'].unique():
        test_results = results_df[results_df['Test'] == test]
        print(f"  {test}: {test_results['P-value'].mean():.4f} (n={len(test_results)})")
    
    print(f"\nEstaciones con distribución MÁS NORMAL (p-valor más alto):")
    top_normal = results_df.nlargest(3, 'P-value')[['Station', 'P-value', 'Skewness', 'Kurtosis']]
    for _, row in top_normal.iterrows():
        print(f"  {row['Station']}: p={row['P-value']:.4f}, Asimetría={row['Skewness']:.3f}, Curtosis={row['Kurtosis']:.3f}")
    
    print(f"\nEstaciones con distribución MENOS NORMAL (p-valor más bajo):")
    bottom_normal = results_df.nsmallest(3, 'P-value')[['Station', 'P-value', 'Skewness', 'Kurtosis']]
    for _, row in bottom_normal.iterrows():
        print(f"  {row['Station']}: p={row['P-value']:.4f}, Asimetría={row['Skewness']:.3f}, Curtosis={row['Kurtosis']:.3f}")
    
    return results_df

def plot_polar_stations(dfs_estaciones_clean, contaminant_idx, wind_dir_idx,
                        contaminant_name="Contaminante", max_stations=15, nrows=3, ncols=5,
                        cmap="Reds"):
    """
    Genera un grid de subplots polares (rosa de contaminación) mostrando la relación entre
    un contaminante y la dirección del viento para múltiples estaciones.
    
    Parámetros:
    -----------
    dfs_estaciones_clean : dict
        Diccionario {nombre_estacion: DataFrame}.
    contaminant_idx : int
        Índice de la columna del contaminante en el DataFrame.
    wind_dir_idx : int
        Índice de la columna de dirección del viento en el DataFrame.
    contaminant_name : str
        Nombre descriptivo del contaminante (ej. "PM2.5").
    max_stations : int
        Número máximo de estaciones a mostrar (ej. 15).
    nrows, ncols : int
        Dimensiones del grid de subplots.
    cmap : str
        Colormap para el scatter.
    
    Retorna:
    --------
    results_df : pd.DataFrame
        Resumen con correlaciones u/v por estación.
    """

    print("🌪️" * 25)
    print(f"ANÁLISIS DE CORRELACIÓN: {contaminant_name} vs DIRECCIÓN DEL VIENTO")
    print("🌪️" * 25)
    print(f"Analizando la relación entre concentraciones de {contaminant_name} y la dirección del viento")
    print("=" * 80)

    # Crear figura
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12), subplot_kw={'projection': 'polar'})
    axes = axes.flatten()

    correlation_results = []

    for i, (station_name, df_station) in enumerate(dfs_estaciones_clean.items()):
        if i >= max_stations:
            break
        
        ax = axes[i]
        
        contaminant_data = df_station.iloc[:, contaminant_idx]
        wind_dir_data = df_station.iloc[:, wind_dir_idx]
        
        # Eliminar NaN
        mask = ~(pd.isna(contaminant_data) | pd.isna(wind_dir_data))
        contaminant_clean = contaminant_data[mask]
        wind_dir_clean = wind_dir_data[mask]
        
        if len(contaminant_clean) > 0:
            # Dirección en radianes
            wind_rad = np.deg2rad(wind_dir_clean.values)
            
            # Componentes trigonométricas
            u = np.cos(wind_rad)  # eje E-O
            v = np.sin(wind_rad)  # eje N-S
            
            # Correlaciones
            corr_u = np.corrcoef(contaminant_clean, u)[0, 1]
            corr_v = np.corrcoef(contaminant_clean, v)[0, 1]
            
            # Scatter polar
            sc = ax.scatter(wind_rad, contaminant_clean, alpha=0.6, s=15,
                            c=contaminant_clean, cmap=cmap)
            
            ax.set_title(f"{station_name}\n r_u={corr_u:.2f}, r_v={corr_v:.2f}",
                         fontsize=9, fontweight='bold')
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            
            correlation_results.append({
                'Station': station_name,
                f'Corr_{contaminant_name}_u': corr_u,
                f'Corr_{contaminant_name}_v': corr_v,
                'N_Points': len(contaminant_clean),
                f'{contaminant_name}_Mean': contaminant_clean.mean()
            })
        else:
            ax.text(0.5, 0.5, 'Sin datos\nsuficientes', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(f'{station_name}\n(Sin datos)', fontsize=10)

    # Título general
    plt.suptitle(f'Relación entre {contaminant_name} y Dirección del Viento (Rosa de Contaminación)\n(Datos Limpios)', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

    # DataFrame de resultados
    results_df = pd.DataFrame(correlation_results)
    return results_df

def analyze_wind_sectors(dfs_estaciones_clean, contaminant_idx, wind_dir_idx,
                         contaminant_name="Contaminante", n_sectors=16, threshold_factor=2.0, 
                         max_stations=15, nrows=3, ncols=5, plot=True):
    """
    Analiza sectores del viento que concentran valores inusualmente altos
    de un contaminante para múltiples estaciones (rosa de contaminación).

    Parámetros
    ----------
    dfs_estaciones_clean : dict
        Diccionario {nombre_estacion: DataFrame}.
    contaminant_idx : int
        Índice de columna del contaminante en el DataFrame.
    wind_dir_idx : int
        Índice de columna de la dirección del viento (grados).
    contaminant_name : str
        Nombre descriptivo del contaminante (ej. "PM2.5").
    n_sectors : int, opcional
        Número de sectores (ej. 16 → sectores de 22.5°).
    threshold_factor : float, opcional
        Factor de comparación (ej. 2.0 → sector interesante si la media sectorial
        es ≥ 2 × la media global).
    max_stations : int
        Número máximo de estaciones a mostrar (ej. 15).
    nrows, ncols : int
        Dimensiones del grid de subplots.
    plot : bool, opcional
        Si True, genera gráfico polar de la rosa de contaminación.

    Retorna
    -------
    list : Lista de diccionarios con estadísticas por estación.
    """

    print("🌹" * 25)
    print(f"ANÁLISIS DE SECTORES DE VIENTO: {contaminant_name}")
    print("🌹" * 25)
    print(f"Identificando sectores con concentraciones altas de {contaminant_name}")
    print(f"Umbral: {threshold_factor}x la media global | Sectores: {n_sectors}")
    print("=" * 80)

    results = []

    if plot:
        # Crear figura con subplots polares
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12), subplot_kw={'projection': 'polar'})
        axes = axes.flatten()

    for i, (station_name, df_station) in enumerate(dfs_estaciones_clean.items()):
        if i >= max_stations:
            break

        contaminant_col = df_station.columns[contaminant_idx]
        wind_col = df_station.columns[wind_dir_idx]

        # Extraer y limpiar datos
        data = df_station[[contaminant_col, wind_col]].dropna()
        
        if data.empty:
            result = {
                "Station": station_name,
                "Contaminant": contaminant_col,
                "GlobalMean": np.nan,
                "InterestingSectors": [],
                "SectorMeans": {},
                "N_Points": 0
            }
            results.append(result)
            
            if plot:
                ax = axes[i]
                ax.text(0.5, 0.5, 'Sin datos\nsuficientes', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{station_name}\n(Sin datos)', fontsize=9)
            continue

        # Definir bins para sectores
        bins = np.linspace(0, 360, n_sectors + 1)
        sector_labels = [f"{int(bins[i])}°-{int(bins[i+1])}°" for i in range(n_sectors)]
        data["Sector"] = pd.cut(data[wind_col] % 360, bins=bins, labels=sector_labels, include_lowest=True)

        # Calcular ultimo cuartil por sector y media global
        sector_means = data.groupby("Sector", observed = False)[contaminant_col].mean()
        global_mean = data[contaminant_col].mean()

        # Detectar sectores interesantes
        interesting_sectors = sector_means[sector_means >= threshold_factor * global_mean].index.tolist()

        # Guardar resultado
        result = {
            "Station": station_name,
            "Contaminant": contaminant_col,
            "GlobalMean": global_mean,
            "InterestingSectors": interesting_sectors,
            "SectorMeans": sector_means.to_dict(),
            "N_Points": len(data)
        }
        results.append(result)

        # Plot polar para esta estación
        if plot:
            ax = axes[i]
            
            # Preparar datos para el plot polar
            theta = np.linspace(0, 2*np.pi, n_sectors, endpoint=False)
            r = sector_means.values
            
            # Colores: rojo para sectores interesantes, azul para normales
            colors = ["red" if label in interesting_sectors else "steelblue" 
                     for label in sector_means.index]
            
            # Crear barras polares
            bars = ax.bar(theta, r, width=2*np.pi/n_sectors, align='center', 
                         alpha=0.7, color=colors)
            
            # Configurar ejes polares
            ax.set_theta_zero_location("N")  # Norte arriba
            ax.set_theta_direction(-1)       # Sentido horario
            
            # Título con información de sectores interesantes
            n_interesting = len(interesting_sectors)
            title = f"{station_name}\n{n_interesting} sectores críticos"
            ax.set_title(title, fontsize=9, fontweight='bold')
            
            # Línea de referencia para el umbral
            ax.axhline(y=threshold_factor * global_mean, color='red', 
                      linestyle='--', alpha=0.5, linewidth=1)

    if plot:
        # Título general
        plt.suptitle(f'Rosa de Sectores Críticos: {contaminant_name}\n'
                     f'(Rojo: ≥{threshold_factor}x media global, Azul: normal)', 
                     fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()

    # Mostrar resumen de sectores interesantes
    interesting_stations = [r for r in results if len(r["InterestingSectors"]) > 0]
    
    print(f"\n🎯 RESUMEN DE ANÁLISIS:")
    print("=" * 80)
    print(f"Estaciones analizadas: {len(results)}")
    print(f"Estaciones con sectores críticos: {len(interesting_stations)}")
    
    if interesting_stations:
        print(f"\n🌟 ESTACIONES CON SECTORES CRÍTICOS (≥{threshold_factor}x media):")
        print("-" * 60)
        for station_data in interesting_stations:
            sectors_str = ", ".join(station_data["InterestingSectors"])
            print(f"📍 {station_data['Station']}: {sectors_str}")
            print(f"   Media global: {station_data['GlobalMean']:.2f}")
            for sector in station_data["InterestingSectors"]:
                sector_mean = station_data["SectorMeans"][sector]
                factor = sector_mean / station_data['GlobalMean']
                print(f"   • {sector}: {sector_mean:.2f} ({factor:.1f}x media)")
            print()
    else:
        print(f"\n✨ No se encontraron sectores críticos con el umbral {threshold_factor}x")
        print("   Considera reducir el threshold_factor para detectar patrones más sutiles")

    return results

def logistic_model_station(
    df,
    contaminant_idx,
    wind_dir_idx=None,
    wind_speed_idx=None,
    rh_idx=None,
    temp_idx=None,
    threshold=None
):
    """
    Ajusta un modelo de regresión logística para predecir
    cuándo la concentración de un contaminante (ej. NO) supera un valor crítico.

    Parámetros
    ----------
    df : DataFrame (estación individual)
    contaminant_idx : int, índice de columna del contaminante
    wind_dir_idx : int or None, índice de columna de dirección del viento (grados)
    wind_speed_idx : int or None, índice de columna de velocidad del viento
    rh_idx : int or None, índice de columna de humedad relativa
    temp_idx : int or None, índice de columna de temperatura
    threshold : float, umbral crítico (si None, se usa percentil 90)

    Retorna
    -------
    modelo : fitted model de statsmodels
    report : clasificación en test set
    """
    contaminant_col = df.columns[contaminant_idx]
    
    # Lista para almacenar las columnas que vamos a usar
    feature_cols = []
    feature_names = []
    
    # Lista para almacenar solo las columnas que no son None
    data_cols = [contaminant_col]
    
    # Agregar variables solo si no son None
    if wind_dir_idx is not None:
        wind_col = df.columns[wind_dir_idx]
        data_cols.append(wind_col)
        # Para dirección del viento, usaremos componentes sin/cos
        feature_names.extend(["Wind_sin", "Wind_cos"])
    
    if wind_speed_idx is not None:
        speed_col = df.columns[wind_speed_idx]
        data_cols.append(speed_col)
        feature_cols.append(speed_col)
        feature_names.append("Wind_Speed")
    
    if rh_idx is not None:
        rh_col = df.columns[rh_idx]
        data_cols.append(rh_col)
        feature_cols.append(rh_col)
        feature_names.append("Humidity")
    
    if temp_idx is not None:
        temp_col = df.columns[temp_idx]
        data_cols.append(temp_col)
        feature_cols.append(temp_col)
        feature_names.append("Temperature")

    # Verificar que tenemos al menos una variable predictora
    if len(data_cols) == 1:  # Solo tenemos el contaminante
        raise ValueError("Debe proporcionar al menos una variable predictora (índice no None)")

    # Extraer datos solo para las columnas que vamos a usar
    data = df[data_cols].dropna()

    if data.empty:
        raise ValueError("No hay datos suficientes en esta estación después de eliminar NaN.")

    # Definir umbral crítico
    if threshold is None:
        threshold = np.percentile(data[contaminant_col], 90)
    print(f"Umbral crítico: {threshold:.3f}")

    data["Critical"] = (data[contaminant_col] >= threshold).astype(int)

    # Construir matriz de características X
    X_data = []
    
    # Codificar dirección del viento si está presente
    if wind_dir_idx is not None:
        wind_rad = np.deg2rad(data[wind_col] % 360)
        X_data.append(np.sin(wind_rad))  # Wind_sin
        X_data.append(np.cos(wind_rad))  # Wind_cos
    
    # Agregar otras variables si están presentes
    for col in feature_cols:
        X_data.append(data[col].values)
    
    # Crear DataFrame de características
    X = pd.DataFrame(np.column_stack(X_data), columns=feature_names, index=data.index)
    y = data["Critical"]

    # Verificar que tenemos variabilidad en la variable objetivo
    if y.nunique() < 2:
        raise ValueError("La variable objetivo no tiene variabilidad (todos los valores son iguales)")

    print(f"Variables utilizadas en el modelo: {feature_names}")
    print(f"Distribución de la variable objetivo: {y.value_counts().to_dict()}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Ajustar modelo logístico
    X_train_const = sm.add_constant(X_train)
    model = sm.Logit(y_train, X_train_const).fit(disp=False)

    # Evaluación
    X_test_const = sm.add_constant(X_test)
    y_pred = (model.predict(X_test_const) >= 0.15).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)

    return model, report

def analyze_logistic_model(model, report):
    """
    Analiza un modelo logístico ajustado con statsmodels.Logit.
    
    Parámetros
    ----------
    model : fitted model
        Modelo entrenado de statsmodels.Logit.
    report : dict
        Reporte de clasificación de sklearn (classification_report con output_dict=True).
    
    Retorna
    -------
    summary_dict : dict
        Resumen con odds ratios, significancia y métricas de clasificación.
    """
    print("\n📊 RESUMEN DEL MODELO LOGÍSTICO")
    print("=" * 80)
    print(model.summary())  # Incluye coeficientes, errores estándar, z y p-values
    
    # --- Odds Ratios ---
    params = model.params
    conf = model.conf_int()
    conf['OR'] = params
    conf.columns = ['2.5%', '97.5%', 'OR']
    odds_ratios = np.exp(conf)
    
    print("\n📈 ODDS RATIOS (exp(coeficientes))")
    print("=" * 80)
    print(odds_ratios)
    
    # --- Reporte de clasificación ---
    print("\n🎯 DESEMPEÑO EN TEST SET")
    print("=" * 80)
    df_report = pd.DataFrame(report).T
    print(df_report)
    
    # --- Interpretación rápida ---
    print("\n🧐 INTERPRETACIÓN DE SIGNIFICANCIA")
    print("=" * 80)
    sig_vars = model.pvalues[model.pvalues < 0.05].index.tolist()
    if sig_vars:
        print(f"Variables con significancia estadística (p < 0.05): {sig_vars}")
    else:
        print("⚠️ Ninguna variable resultó significativa al nivel 0.05")
    
    # Guardar resumen en dict
    summary_dict = {
        "odds_ratios": odds_ratios,
        "classification_report": df_report,
        "significant_vars": sig_vars
    }
    return summary_dict

def recalibrate_threshold(y_true, y_probs, step=0.05):
    """
    Explora diferentes umbrales de decisión para un modelo logístico.
    
    Parámetros
    ----------
    y_true : array-like
        Valores reales (0/1).
    y_probs : array-like
        Probabilidades predichas por el modelo.
    step : float
        Tamaño del paso en el barrido de umbrales.
        
    Retorna
    -------
    results_df : DataFrame
        Resultados con precision, recall, f1 para cada umbral.
    """
    import pandas as pd
    
    thresholds = np.arange(0.05, 1.0, step)
    metrics = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        metrics.append([t, precision, recall, f1])

    results_df = pd.DataFrame(metrics, columns=["Threshold", "Precision", "Recall", "F1"])
    return results_df

def plot_recalibration(results_df):
    """
    Grafica precision, recall y F1-score contra el umbral de decisión.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["Threshold"], results_df["Precision"], label="Precision", marker="o")
    plt.plot(results_df["Threshold"], results_df["Recall"], label="Recall", marker="o")
    plt.plot(results_df["Threshold"], results_df["F1"], label="F1-score", marker="o")

    plt.axvline(0.5, color="red", linestyle="--", label="Umbral clásico 0.5")
    plt.xlabel("Umbral de decisión")
    plt.ylabel("Métrica")
    plt.title("Recalibración del Umbral de Decisión (Modelo Logístico)")
    plt.legend()
    plt.grid(True)
    plt.show()






