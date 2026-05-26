import datetime
import hashlib
import hmac
import json
import os
import time
from urllib.parse import urlencode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from dotenv import load_dotenv
from google.colab import userdata  # Mantenha se for rodar no Colab, ou mude para os.environ se for local
from openai import OpenAI

# =====================================================================
# CONFIGURAÇÕES E INICIALIZAÇÃO
# =====================================================================

load_dotenv()

# Recuperação de chaves (ajuste para os.getenv("NOME") se rodar 100% localmente)
OPENROUTER_KEY = userdata.get('OPENROUTER_KEY')
FOXBIT_KEY = userdata.get('FOXBIT_KEY')
FOXBIT_SECRET = userdata.get('FOXBIT_SECRET')

API_BASE = "https://api.foxbit.com.br"

# Configura cliente para OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY
)

# =====================================================================
# FUNÇÕES DA API FOXBIT (AUTENTICAÇÃO E REQUISIÇÕES)
# =====================================================================

def gerar_assinatura(api_secret, method, path, params=None, body=""):
    """Gera a assinatura HMAC e cabeçalhos exigidos pela Foxbit."""
    timestamp = str(int(time.time() * 1000))
    queryString = urlencode(params) if params else ''
    rawBody = json.dumps(body) if body else ''
    
    preHash = f"{timestamp}{method.upper()}{path}{queryString}{rawBody}"
    
    assinatura = hmac.new(
        api_secret.encode(), preHash.encode(), hashlib.sha256
    ).hexdigest()
    
    headers = {
        "X-FB-ACCESS-KEY": FOXBIT_KEY,
        "X-FB-ACCESS-TIMESTAMP": timestamp,
        "X-FB-ACCESS-SIGNATURE": assinatura,
        "Content-Type": "application/json",
    }
    return headers


def chamada_api_privada(method, endpoint, payload=None, params=None):
    """Executa chamadas privadas na API da Foxbit."""
    url = f"{API_BASE}{endpoint}"
    # Corrigido de API_SECRET para FOXBIT_SECRET para manter consistência
    headers = gerar_assinatura(FOXBIT_SECRET, method, endpoint, params, payload)
    
    try:
        response = requests.request(method, url, headers=headers, json=payload, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro ao acessar API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print("📩 Resposta da API:", e.response.text)
        return None


def obter_ticker(symbol):
    """Obtém o preço atual de mercado (último preço) para o par informado."""
    endpoint = f"/rest/v3/markets/{symbol}/ticker/24hr"
    try:
        r = requests.get(API_BASE + endpoint)
        r.raise_for_status()
        data = r.json().get("data", {})
        return pd.json_normalize(data)
    except Exception as e:
        print("❌ Erro ao consultar ticker:", e)
        return None


def pegar_candlesticks(symbol, interval, limit):
    """Busca o histórico de candlesticks da Foxbit."""
    endpoint = f"/rest/v3/markets/{symbol}/candlesticks"
    params = {"interval": interval, "limit": limit}
    
    response = requests.get(API_BASE + endpoint, params=params)
    if response.status_code != 200:
        print(f"❌ Erro ao obter candlesticks. Status code: {response.status_code}")
        print("Resposta bruta:", response.text)
        raise Exception("Erro na API ao obter candlesticks.")
        
    candles = response.json()
    
    df = pd.DataFrame(candles, columns=[
        "timestamp_open", "open", "high", "low", "close",
        "timestamp_close", "volume", "quoteVolume", "count",
        "takerBuyVolume", "takerBuyQuoteVolume"
    ])
    
    df["timestamp_open"] = pd.to_datetime(df["timestamp_open"].astype(int), unit='ms')
    df["timestamp_close"] = pd.to_datetime(df["timestamp_close"].astype(int), unit='ms')
    
    return df


def obter_orderbook(symbol):
    """Obtém o livro de ordens (orderbook) para o par informado."""
    endpoint = f"/rest/v3/markets/{symbol}/orderbook"
    try:
        r = requests.get(API_BASE + endpoint)
        r.raise_for_status()
        data = r.json()
        
        df_asks = pd.DataFrame(data['asks'], columns=['price', 'volume'])
        df_asks['type'] = 'ask'
        
        df_bids = pd.DataFrame(data['bids'], columns=['price', 'volume'])
        df_bids['type'] = 'bid'
        
        return pd.concat([df_asks, df_bids], ignore_index=True)
    except Exception as e:
        print("❌ Erro ao consultar orderbook:", e)
        return None

# =====================================================================
# PROCESSAMENTO DE DADOS E MICROESTRUTURA DE MERCADO
# =====================================================================

def resumir_candles(df, n=20):
    """Calcula indicadores técnicos (RSI, SMA, Tendência) com base nos candles."""
    df = df.tail(n).copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    
    if len(df) < 2:
        return {
            "preco_atual": None, "retorno_curto_pct": None, "volatilidade_pct": None,
            "tendencia_curta": None, "rsi_14": None, "distancia_sma_20_pct": None
        }
        
    preco_atual = df.iloc[-1]["close"]
    preco_inicial = df.iloc[0]["close"]
    
    retorno_pct = ((preco_atual - preco_inicial) / preco_inicial) * 100
    volatilidade_pct = df["close"].pct_change().std() * 100
    
    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)
    rsi_atual = rsi.iloc[-1]
    
    # SMA 20
    sma_20 = df["close"].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else np.nan
    distancia_sma_pct = ((preco_atual - sma_20) / sma_20) * 100 if not pd.isna(sma_20) else None
    
    # Tendência
    if retorno_pct > 0.2:
        tendencia = "alta"
    elif retorno_pct < -0.2:
        tendencia = "baixa"
    else:
        tendencia = "lateral"
        
    return {
        "preco_atual": round(float(preco_atual), 4),
        "retorno_curto_pct": round(float(retorno_pct), 3),
        "volatilidade_pct": None if pd.isna(volatilidade_pct) else round(float(volatilidade_pct), 3),
        "tendencia_curta": tendencia,
        "rsi_14": None if pd.isna(rsi_atual) else round(float(rsi_atual), 2),
        "distancia_sma_20_pct": None if distancia_sma_pct is None or pd.isna(distancia_sma_pct) else round(float(distancia_sma_pct), 3)
    }


def resumir_ticker_df(df_ticker):
    """Extrai informações resumidas do DataFrame do Ticker."""
    last = df_ticker.iloc[-1]
    last_price = float(last["last_trade.price"])
    bid_price = float(last["best.bid.price"])
    ask_price = float(last["best.ask.price"])
    
    spread_pct = ((ask_price - bid_price) / last_price) * 100
    
    return {
        "preco_atual": last_price,
        "spread_pct": round(spread_pct, 4),
        "variacao_24h_pct": float(last["rolling_24h.price_change_percent"]),
        "volume_24h": float(last["rolling_24h.volume"]),
        "trades_24h": int(last["rolling_24h.trades_count"]),
        "max_24h": float(last["rolling_24h.high"]),
        "min_24h": float(last["rolling_24h.low"])
    }


def pressao_bid_ask_df(df_ticker):
    """Calcula a proporção direta entre o volume do melhor bid e do melhor ask."""
    last = df_ticker.iloc[-1]
    bid_vol = float(last["best.bid.volume"])
    ask_vol = float(last["best.ask.volume"])
    razao = None if ask_vol == 0 else bid_vol / ask_vol
    return {"bid_ask_ratio_top": None if razao is None else round(razao, 3)}


def resumir_orderbook_df(df_orderbook, top_n=5):
    """Resume a pressão de compra/venda bruta no topo do livro de ordens."""
    asks = df_orderbook[df_orderbook["type"] == "ask"].head(top_n)
    bids = df_orderbook[df_orderbook["type"] == "bid"].head(top_n)
    
    vol_asks = asks["volume"].astype(float).sum()
    vol_bids = bids["volume"].astype(float).sum()
    pressao = None if vol_asks == 0 else vol_bids / vol_asks
    
    return {
        "volume_asks_top": round(float(vol_asks), 6),
        "volume_bids_top": round(float(vol_bids), 6),
        "pressao_compra": None if pressao is None else round(float(pressao), 3)
    }


def resumir_orderbook_ponderado(df_orderbook, preco_atual, top_n=10):
    """Calcula a pressão de compra ponderada pela distância do preço atual."""
    df = df_orderbook.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    
    asks = df[df["type"] == "ask"].head(top_n)
    bids = df[df["type"] == "bid"].head(top_n)
    
    asks["peso"] = preco_atual / asks["price"].replace(0, np.nan)
    bids["peso"] = bids["price"] / preco_atual
    
    volume_asks_ponderado = (asks["volume"] * asks["peso"]).sum()
    volume_bids_ponderado = (bids["volume"] * bids["peso"]).sum()
    total = volume_asks_ponderado + volume_bids_ponderado
    
    return {"pressao_compra_ponderada": (volume_bids_ponderado / total if total > 0 else None)}


def montar_contexto_mercado(df_candles, df_ticker, df_orderbook):
    """Agrega todas as métricas de mercado calculadas em um único dicionário estruturado."""
    contexto = {}
    contexto.update(resumir_candles(df_candles))
    contexto.update(resumir_ticker_df(df_ticker))
    contexto.update(pressao_bid_ask_df(df_ticker))
    contexto.update(resumir_orderbook_df(df_orderbook))
    
    if "preco_atual" in contexto:
        contexto.update(resumir_orderbook_ponderado(df_orderbook, contexto["preco_atual"]))
    else:
        print("⚠️ 'preco_atual' not found in context. Cannot calculate 'pressao_compra_ponderada'.")
        
    contexto["ativo"] = "BTC/BRL"
    contexto["timeframe"] = "1m"
    return contexto

# =====================================================================
# TOMADA DE DECISÃO COM IA
# =====================================================================

def montar_prompt_ia(contexto):
    """Gera a string estruturada do Prompt que alimenta o LLM."""
    return f"""
Você é um analista quantitativo especializado em mercado financeiro, microestrutura e fluxo de ordens.
Avalie o estado atual do mercado e decida a melhor ação entre: COMPRAR, VENDER ou ESPERAR.

Contexto atual do mercado:
Ativo: {contexto['ativo']}
Timeframe: {contexto['timeframe']}

Preço atual: {contexto['preco_atual']}
Retorno curto (%): {contexto['retorno_curto_pct']}
Volatilidade (%): {contexto['volatilidade_pct']}
Tendência curta: {contexto['tendencia_curta']}

Variação 24h (%): {contexto['variacao_24h_pct']}
Máxima 24h: {contexto['max_24h']}
Mínima 24h: {contexto['min_24h']}
Volume 24h: {contexto['volume_24h']}
Trades 24h: {contexto['trades_24h']}

Spread (%): {contexto['spread_pct']}

Order Book:
- Volume bids (top): {contexto['volume_bids_top']}
- Volume asks (top): {contexto['volume_asks_top']}
- Bid/Ask ratio: {contexto['bid_ask_ratio_top']}
- Pressão de compra: {contexto['pressao_compra']}
- Pressão de compra ponderada: {contexto['pressao_compra_ponderada']}

Responda exclusivamente no formato JSON abaixo:
{{
  "acao": "COMPRAR | VENDER | ESPERAR",
  "confianca": 0.0,
  "justificativa_curta": ""
}}
"""


def analisar_mercado(contexto_mercado, openai_client):
    """Envia o contexto consolidado de mercado para a IA via OpenRouter."""
    prompt_final = montar_prompt_ia(contexto_mercado)
    try:
        response = openai_client.chat.completions.create(
            model="deepseek/deepseek-v4-flash:free",
            messages=[{"role": "user", "content": prompt_final}],
            max_tokens=300,
            temperature=0.2
        )
        if response and response.choices and response.choices[0].message:
            return response.choices[0].message.content
        else:
            print(f"❌ API response did not contain expected choices or message: {response}")
            return json.dumps({"acao": "ESPERAR", "confianca": 0.0, "justificativa_curta": "API response missing choices."})
    except Exception as e:
        print(f"❌ Erro ao chamar a API da IA: {e}")
        return json.dumps({"acao": "ESPERAR", "confianca": 0.0, "justificativa_curta": f"Erro na chamada da API da IA: {e}"})


def analisar_mercado_wrapper(contexto_mercado):
    return analisar_mercado(contexto_mercado, client)

# =====================================================================
# GERENCIAMENTO DE TRANSAÇÕES E PORTFÓLIO
# =====================================================================

def gerenciar_transacoes(acao, contexto_mercado, portfolio_saldo, confianca_ia, max_capital_por_transacao_brl=10, max_pct_btc_a_vender=0.5):
    """Simula a execução física de ordens e atualiza os saldos financeiros do portfólio."""
    timestamp = datetime.datetime.now().isoformat()
    preco_atual = contexto_mercado['preco_atual']
    ativo = contexto_mercado['ativo']
    brl_disponivel = portfolio_saldo.get('brl', 0.0)
    btc_disponivel = portfolio_saldo.get('btc', 0.0)
    transacoes = portfolio_saldo.get('transacoes', [])
    
    capital_a_alocar_brl = max_capital_por_transacao_brl * confianca_ia
    
    if acao == "COMPRAR":
        if brl_disponivel >= capital_a_alocar_brl and capital_a_alocar_brl > 0.01:
            quantidade_btc = capital_a_alocar_brl / preco_atual
            brl_disponivel -= capital_a_alocar_brl
            btc_disponivel += quantidade_btc
            transacao = {
                'timestamp': timestamp, 'ativo': ativo, 'acao': 'COMPRAR', 'preco_btc': preco_atual,
                'quantidade_btc': quantidade_btc, 'valor_brl': capital_a_alocar_brl, 'confianca_ia': confianca_ia
            }
            transacoes.append(transacao)
            print(f"✅ COMPRA realizada: {quantidade_btc:.6f} {ativo} a {preco_atual:.2f} BRL. Alocado: {capital_a_alocar_brl:.2f} BRL.")
        else:
            print("🛑 Saldo em BRL insuficiente ou confiança da IA muito baixa para COMPRAR.")
            transacoes.append({'timestamp': timestamp, 'ativo': ativo, 'acao': 'COMPRAR_FALHA', 'motivo': 'Saldo insuficiente/baixa confiança', 'preco_btc': preco_atual, 'confianca_ia': confianca_ia})
            
    elif acao == "VENDER":
        quantidade_btc_a_vender = btc_disponivel * (max_pct_btc_a_vender * confianca_ia)
        if quantidade_btc_a_vender > 0.00000001:
            valor_brl_venda = quantidade_btc_a_vender * preco_atual
            brl_disponivel += valor_brl_venda
            btc_disponivel -= quantidade_btc_a_vender
            
            compras_validas = [t for t in transacoes if t['acao'] == 'COMPRAR']
            custo_total_btc = sum(t['valor_brl'] for t in compras_validas)
            total_btc_comprado = sum(t['quantidade_btc'] for t in compras_validas)
            
            lucro_brl = 0
            if total_btc_comprado > 0:
                custo_medio_por_btc = custo_total_btc / total_btc_comprado
                lucro_brl = (preco_atual - custo_medio_por_btc) * quantidade_btc_a_vender
                
            transacao = {
                'timestamp': timestamp, 'ativo': ativo, 'acao': 'VENDER', 'preco_btc': preco_atual,
                'quantidade_btc': quantidade_btc_a_vender, 'valor_brl_venda': valor_brl_venda, 'lucro_brl_estimado': lucro_brl, 'confianca_ia': confianca_ia
            }
            transacoes.append(transacao)
            print(f"💰 VENDA realizada: {quantidade_btc_a_vender:.6f} {ativo} a {preco_atual:.2f} BRL. Lucro/Prejuízo: {lucro_brl:.2f} BRL.")
        else:
            print("🛑 Nenhum BTC para VENDER ou confiança da IA muito baixa.")
            transacoes.append({'timestamp': timestamp, 'ativo': ativo, 'acao': 'VENDER_FALHA', 'motivo': 'Sem ativos/baixa confiança', 'preco_btc': preco_atual, 'confianca_ia': confianca_ia})
            
    elif acao == "ESPERAR":
        print("⏳ ESPERAR: Nenhuma ação tomada.")
        transacoes.append({'timestamp': timestamp, 'ativo': ativo, 'acao': 'ESPERAR', 'preco_btc': preco_atual, 'confianca_ia': confianca_ia})
        
    portfolio_saldo['brl'] = brl_disponivel
    portfolio_saldo['btc'] = btc_disponivel
    portfolio_saldo['transacoes'] = transacoes
    return portfolio_saldo

# =====================================================================
# VISUALIZAÇÃO DE RESULTADOS
# =====================================================================

def plotar_resultados(portfolio_history_list):
    """Gera os gráficos de desempenho financeiro baseados na lista de histórico do portfólio."""
    if not portfolio_history_list:
        print("⚠️ Sem dados de histórico para plotar gráficos.")
        return

    df_portfolio = pd.DataFrame(portfolio_history_list)
    df_portfolio['timestamp'] = pd.to_datetime(df_portfolio['timestamp'])
    
    # Gráfico 1: Evolução total em BRL
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='timestamp', y='total_value_brl', data=df_portfolio)
    plt.title('Evolução do Valor Total do Portfólio (BRL)')
    plt.xlabel('Tempo')
    plt.ylabel('Valor Total em BRL')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Gráfico 2: Evolução individual de Balanços
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    sns.lineplot(x='timestamp', y='brl_balance', data=df_portfolio, ax=axes[0], color='blue')
    axes[0].set_title('Evolução do Saldo em BRL')
    axes[0].set_ylabel('BRL')
    axes[0].grid(True)
    
    sns.lineplot(x='timestamp', y='btc_balance', data=df_portfolio, ax=axes[1], color='orange')
    axes[1].set_title('Evolução do Saldo em BTC')
    axes[1].set_xlabel('Tempo')
    axes[1].set_ylabel('BTC')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# =====================================================================
# LOOP PRINCIPAL DA SIMULAÇÃO (EXECUÇÃO)
# =====================================================================

if __name__ == "__main__":
    # Inicialização do Portfólio de Testes
    portfolio = {'brl': 100.0, 'btc': 0.0, 'transacoes': []}
    portfolio_history = []
    
    MAX_CAPITAL_POR_TRANSACAO_BRL = 10.0
    MAX_PCT_BTC_A_VENDER = 0.5
    NUM_OPERACOES = 10
    PROFIT_TARGET_PCT = 2.0
    
    print(f"Portfólio inicial da simulação: {portfolio['brl']:.2f} BRL, {portfolio['btc']:.6f} BTC")
    
    for i in range(NUM_OPERACOES):
        print(f"\n--- Simulação de Operação {i+1}/{NUM_OPERACOES} ---")
        contexto_mercado = None
        
        # 1. Obter e Estruturar Dados do Mercado
        try:
            df_candles = pegar_candlesticks("btcbrl", "1m", "10")
            df_ticker = obter_ticker("btcbrl")
            df_orderbook = obter_orderbook("btcbrl")
            
            if df_candles is None or df_ticker is None or df_orderbook is None:
                print("⚠️ Erro ao obter dados de mercado. Pulando esta iteração.")
                time.sleep(5)
                continue
                
            contexto_mercado = montar_contexto_mercado(df_candles, df_ticker, df_orderbook)
        except Exception as e:
            print(f"⚠️ Erro ao montar contexto de mercado: {e}. Pulando esta iteração.")
            time.sleep(5)
            continue
            
        # 2. Consultar Decisão Estratégica da IA
        decisao_str = analisar_mercado_wrapper(contexto_mercado)
        confianca_ia = 0.0
        acao_ia = "ESPERAR"
        
        try:
            decisao_ia_dict = json.loads(decisao_str)
            acao_ia = decisao_ia_dict['acao']
            confianca_ia = decisao_ia_dict.get('confianca', 0.0)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"❌ Erro ao decodificar JSON da IA. Usando 'ESPERAR'. Erro: {e}")
            
        acao_final = acao_ia
        
        # 3. Regra Analítica de Venda por Alvo de Lucro Fixo (2%)
        if portfolio['btc'] > 0:
            compras_validas = [t for t in portfolio['transacoes'] if t['acao'] == 'COMPRAR']
            if compras_validas:
                custo_total_btc = sum(t['valor_brl'] for t in compras_validas)
                total_btc_comprado = sum(t['quantidade_btc'] for t in compras_validas)
                
                if total_btc_comprado > 0:
                    custo_medio_por_btc = custo_total_btc / total_btc_comprado
                    preco_atual = contexto_mercado.get('preco_atual')
                    
                    if preco_atual and preco_atual >= custo_medio_por_btc * (1 + PROFIT_TARGET_PCT / 100):
                        acao_final = "VENDER"
                        confianca_ia = 1.0  # Força a execução total
                        print(f"🎯 Condição de LUCRO de {PROFIT_TARGET_PCT}% atingida! Forçando VENDER.")
                    else:
                        print(f"🔍 Alvo de {PROFIT_TARGET_PCT}% não atingido (Atual: {preco_atual:.2f} / Médio: {custo_medio_por_btc:.2f}).")
                        
        # 4. Executar a Transação no Portfólio
        print(f"IA recomendou: {acao_ia} ({confianca_ia:.2f}) -> Executando: {acao_final}")
        portfolio = gerenciar_transacoes(
            acao_final, contexto_mercado, portfolio, confianca_ia,
            max_capital_por_transacao_brl=MAX_CAPITAL_POR_TRANSACAO_BRL,
            max_pct_btc_a_vender=MAX_PCT_BTC_A_VENDER
        )
        
        # Guardar Histórico temporal da rodada
        current_price = contexto_mercado.get('preco_atual', 0.0)
        total_value_brl = portfolio['brl'] + (portfolio['btc'] * current_price)
        portfolio_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'brl_balance': portfolio['brl'],
            'btc_balance': portfolio['btc'],
            'btc_price_at_step': current_price,
            'total_value_brl': total_value_brl
        })
        
        time.sleep(5)
        
    print("\n--- Simulação Concluída ---")
    print(f"Portfólio final: BRL={portfolio['brl']:.2f}, BTC={portfolio['btc']:.6f}")
    
    # Renderizar Gráficos Finais
    plotar_resultados(portfolio_history)