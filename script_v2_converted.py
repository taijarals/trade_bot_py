# ----------------------------------------------------------------------# Código da célulaimport os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import json
import requests
import pandas as pd
#from foxbit_api import API_BASE # Importa a base da API
import time
import hmac
import hashlib
import json
import requests
from urllib.parse import urlencode
import numpy as np
from openai import OpenAI


from google.colab import userdata
OPENROUTER_KEY = userdata.get('OPENROUTER_KEY')

# Carrega chave de ambiente
load_dotenv()

# Configura cliente para OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY
)
# ----------------------------------------------------------------------# Código da célulafrom google.colab import userdata
FOXBIT_KEY = userdata.get('FOXBIT_KEY')
FOXBIT_SECRET = userdata.get('FOXBIT_SECRET')

API_BASE = "https://api.foxbit.com.br"



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
    url = f"{API_BASE}{endpoint}"

    headers = gerar_assinatura(API_SECRET, method, endpoint, params, payload)

    try:
        response = requests.request(method, url, headers=headers, json=payload, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro ao acessar API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print("📩 Resposta da API:", e.response.text)
        return None

# ----------------------------------------------------------------------# Código da céluladef obter_ticker(symbol):
    """Obtém o preço atual de mercado (último preço) para o par informado."""
    endpoint = f"/rest/v3/markets/{symbol}/ticker/24hr"
    try:
        r = requests.get(API_BASE + endpoint)
        r.raise_for_status()
        data = r.json().get("data", {})

        df_ticker = pd.json_normalize(data)

        return df_ticker

    except Exception as e:
        print("❌ Erro ao consultar ticker:", e)
        return None


def pegar_candlesticks(symbol, interval, limit):
    endpoint = f"/rest/v3/markets/{symbol}/candlesticks"
    params = {"interval": interval, "limit": limit}

    response = requests.get(API_BASE + endpoint, params=params)
    if response.status_code != 200:
        print(f"❌ Erro ao obter candlesticks. Status code: {response.status_code}")
        print("Resposta bruta:", response.text)
        raise Exception("Erro na API ao obter candlesticks.")

    candles = response.json() # lista de listas

    df = pd.DataFrame(candles, columns=[
        "timestamp_open", "open", "high", "low", "close",
        "timestamp_close", "volume", "quoteVolume", "count",
        "takerBuyVolume", "takerBuyQuoteVolume"
    ])

    # Converter timestamps para datetime legível
    # Corrigindo FutureWarning: explicitar o cast para int
    df["timestamp_open"] = pd.to_datetime(df["timestamp_open"].astype(int), unit='ms')
    df["timestamp_close"] = pd.to_datetime(df["timestamp_close"].astype(int), unit='ms')

    return df


def obter_orderbook(symbol):
    """Obtém o livro de ordens (orderbook) para o par informado."""
    endpoint = f"/rest/v3/markets/{symbol}/orderbook"

    params = {"level": 5}

    try:
        r = requests.get(API_BASE + endpoint)
        r.raise_for_status()
        data = r.json()

        # Transformando asks em DataFrame
        df_asks = pd.DataFrame(data['asks'], columns=['price', 'volume'])
        df_asks['type'] = 'ask'

        # Transformando bids em DataFrame
        df_bids = pd.DataFrame(data['bids'], columns=['price', 'volume'])
        df_bids['type'] = 'bid'

        df_order_book = pd.concat([df_asks, df_bids], ignore_index=True)

        return df_order_book

    except Exception as e:
        print("❌ Erro ao consultar orderbook:", e)
        return None
# ----------------------------------------------------------------------# Código da céluladf_candles = pegar_candlesticks("btcbrl", "1m", "10")
df_ticker = obter_ticker("btcbrl")
df_orderbook = obter_orderbook("btcbrl")
# ----------------------------------------------------------------------# Código da céluladef resumir_candles(df, n=20):
    import pandas as pd
    import numpy as np

    df = df.tail(n).copy()

    # garante tipo numérico
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # remove possíveis NaNs
    df = df.dropna(subset=["close"])

    # validação mínima
    if len(df) < 2:
        return {
            "preco_atual": None,
            "retorno_curto_pct": None,
            "volatilidade_pct": None,
            "tendencia_curta": None,
            "rsi_14": None,
            "distancia_sma_20_pct": None
        }

    # =========================
    # PREÇOS
    # =========================
    preco_atual = df.iloc[-1]["close"]
    preco_inicial = df.iloc[0]["close"]

    retorno_pct = ((preco_atual - preco_inicial) / preco_inicial) * 100

    volatilidade_pct = df["close"].pct_change().std() * 100

    # =========================
    # RSI 14
    # =========================
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    # evita divisão por zero sem perder pandas.Series
    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    # quando não houver perdas, RSI tende a 100
    rsi = rsi.fillna(100)

    rsi_atual = rsi.iloc[-1]

    # =========================
    # SMA 20
    # =========================
    sma_20 = (
        df["close"].rolling(window=20).mean().iloc[-1]
        if len(df) >= 20
        else np.nan
    )

    distancia_sma_pct = None

    if not pd.isna(sma_20):
        distancia_sma_pct = (
            ((preco_atual - sma_20) / sma_20) * 100
        )

    # =========================
    # TENDÊNCIA
    # =========================
    if retorno_pct > 0.2:
        tendencia = "alta"

    elif retorno_pct < -0.2:
        tendencia = "baixa"

    else:
        tendencia = "lateral"

    # =========================
    # RETORNO
    # =========================
    return {
        "preco_atual": round(float(preco_atual), 4),

        "retorno_curto_pct": round(float(retorno_pct), 3),

        "volatilidade_pct": (
            None
            if pd.isna(volatilidade_pct)
            else round(float(volatilidade_pct), 3)
        ),

        "tendencia_curta": tendencia,

        "rsi_14": (
            None
            if pd.isna(rsi_atual)
            else round(float(rsi_atual), 2)
        ),

        "distancia_sma_20_pct": (
            None
            if distancia_sma_pct is None or pd.isna(distancia_sma_pct)
            else round(float(distancia_sma_pct), 3)
        )
    }
# ----------------------------------------------------------------------# Código da célula

def resumir_ticker_df(df_ticker):
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
    last = df_ticker.iloc[-1]

    bid_vol = float(last["best.bid.volume"])
    ask_vol = float(last["best.ask.volume"])

    if ask_vol == 0:
        razao = None
    else:
        razao = bid_vol / ask_vol

    return {
        "bid_ask_ratio_top": None if razao is None else round(razao, 3)
    }

def resumir_orderbook_df(df_orderbook, top_n=5):
    asks = df_orderbook[df_orderbook["type"] == "ask"].head(top_n)
    bids = df_orderbook[df_orderbook["type"] == "bid"].head(top_n)

    vol_asks = asks["volume"].astype(float).sum()
    vol_bids = bids["volume"].astype(float).sum()

    if vol_asks == 0:
        pressao = None
    else:
        pressao = vol_bids / vol_asks

    return {
        "volume_asks_top": round(float(vol_asks), 6),
        "volume_bids_top": round(float(vol_bids), 6),
        "pressao_compra": None if pressao is None else round(float(pressao), 3)
    }

def resumir_orderbook_ponderado(df_orderbook, preco_atual, top_n=10):
    df = df_orderbook.copy()

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    asks = df[df["type"] == "ask"].head(top_n)
    bids = df[df["type"] == "bid"].head(top_n)

    # Evitar divisão por zero ou valores não numéricos em 'price'
    asks["peso"] = preco_atual / asks["price"].replace(0, np.nan) # replace 0 with NaN to avoid division by zero
    bids["peso"] = bids["price"] / preco_atual

    volume_asks_ponderado = (asks["volume"] * asks["peso"]).sum()
    volume_bids_ponderado = (bids["volume"] * bids["peso"]).sum()

    total = volume_asks_ponderado + volume_bids_ponderado

    return {
        "pressao_compra_ponderada": (
            volume_bids_ponderado / total if total > 0 else None
        )
    }

def montar_contexto_mercado(df_candles, df_ticker, df_orderbook):
    contexto = {}

    contexto.update(resumir_candles(df_candles))
    contexto.update(resumir_ticker_df(df_ticker))
    contexto.update(pressao_bid_ask_df(df_ticker))
    contexto.update(resumir_orderbook_df(df_orderbook))
    # Check if 'preco_atual' exists before calling resumir_orderbook_ponderado
    if "preco_atual" in contexto:
        contexto.update(resumir_orderbook_ponderado(df_orderbook, contexto["preco_atual"])) # Pass preco_atual
    else:
        print("""⚠️ 'preco_atual' not found in context. Cannot calculate 'pressao_compra_ponderada'.""")

    contexto["ativo"] = "BTC/BRL"
    contexto["timeframe"] = "1m"

    return contexto

# ----------------------------------------------------------------------# Código da célulacontexto_mercado = montar_contexto_mercado(
    df_candles,
    df_ticker,
    df_orderbook
)

contexto_mercado
# ----------------------------------------------------------------------# Código da célula
# ==============================================================
# 🧠 INTELIGÊNCIA ARTIFICIAL
# ==============================================================

def montar_prompt_ia(contexto):
    prompt = f"""
Você é um analista quantitativo especializado em mercado financeiro,
microestrutura e fluxo de ordens.

Avalie o estado atual do mercado e decida a melhor ação entre:
COMPRAR, VENDER ou ESPERAR.

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
    return prompt

def analisar_mercado(contexto_mercado, client):
    prompt_final = montar_prompt_ia(contexto_mercado)

    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-v4-flash:free",
            messages=[
                {"role": "user", "content": prompt_final}
            ],
            max_tokens=300,
            temperature=0.2
        )

        if response and response.choices and response.choices[0].message:
            return response.choices[0].message.content
        else:
            # Log the full response object for debugging if choices are missing
            print(f"❌ API response did not contain expected choices or message: {response}")
            return json.dumps({"acao": "ESPERAR", "confianca": 0.0, "justificativa_curta": "API response missing choices or message."})
    except Exception as e:
        print(f"❌ Erro ao chamar a API da IA: {e}")
        return json.dumps({"acao": "ESPERAR", "confianca": 0.0, "justificativa_curta": f"Erro na chamada da API da IA: {e}"})

# ----------------------------------------------------------------------# Código da céluladef analisar_mercado_wrapper(contexto_mercado):
    resposta_texto = analisar_mercado(contexto_mercado, client)
    return resposta_texto
# ----------------------------------------------------------------------# Código da céluladecisao = analisar_mercado_wrapper(contexto_mercado)
print(decisao)
# ----------------------------------------------------------------------# ### Gerenciamento de Transações (Fluxo de Caixa / Portfólio)# # A função `gerenciar_transacoes` simula um sistema básico de fluxo de caixa/portfólio. Ela recebe a ação da IA ('COMPRAR', 'VENDER', 'ESPERAR'), o contexto de mercado atual e o estado atual do portfólio (saldos em BRL e BTC, e histórico de transações).# # - Se a ação for 'COMPRAR', ela tentará comprar uma quantidade fixa de BTC (baseada em um valor em BRL definido, por exemplo, 1000 BRL) e atualizará os saldos.# - Se a ação for 'VENDER', ela tentará vender todo o BTC disponível e calculará um lucro/prejuízo estimado.# - Se a ação for 'ESPERAR', nenhuma transação será realizada.
# ----------------------------------------------------------------------# Código da célulaimport datetime
import json

def gerenciar_transacoes(acao, contexto_mercado, portfolio_saldo, confianca_ia, max_capital_por_transacao_brl=10, max_pct_btc_a_vender=0.5):
    """
    Gerencia as transações de compra e venda com base na ação e confiança da IA.

    Args:
        acao (str): Ação a ser tomada ('COMPRAR', 'VENDER', 'ESPERAR').
        contexto_mercado (dict): Contexto atual do mercado (contém 'preco_atual', 'ativo').
        portfolio_saldo (dict): Dicionário com saldo atual de BRL, BTC e histórico de transações.
                                Ex: {'brl': 10000.0, 'btc': 0.0, 'transacoes': []}
        confianca_ia (float): Pontuação de confiança da IA na sua decisão (0.0 a 1.0).
        max_capital_por_transacao_brl (float): O capital MÁXIMO em BRL a ser usado por transação de compra, que será ponderado pela confiança.
        max_pct_btc_a_vender (float): O percentual MÁXIMO do BTC a ser vendido, que será ponderado pela confiança.

    Returns:
        dict: O portfolio_saldo atualizado.
    """
    timestamp = datetime.datetime.now().isoformat()
    preco_atual = contexto_mercado['preco_atual']
    ativo = contexto_mercado['ativo']
    brl_disponivel = portfolio_saldo.get('brl', 0.0)
    btc_disponivel = portfolio_saldo.get('btc', 0.0)
    transacoes = portfolio_saldo.get('transacoes', [])

    # Ponderar o capital/quantidade pela confiança da IA
    capital_a_alocar_brl = max_capital_por_transacao_brl * confianca_ia

    if acao == "COMPRAR":
        if brl_disponivel >= capital_a_alocar_brl and capital_a_alocar_brl > 0.01: # Mínimo para evitar compras insignificantes
            quantidade_btc = capital_a_alocar_brl / preco_atual
            brl_disponivel -= capital_a_alocar_brl
            btc_disponivel += quantidade_btc
            transacao = {
                'timestamp': timestamp,
                'ativo': ativo,
                'acao': 'COMPRAR',
                'preco_btc': preco_atual,
                'quantidade_btc': quantidade_btc,
                'valor_brl': capital_a_alocar_brl,
                'confianca_ia': confianca_ia
            }
            transacoes.append(transacao)
            print(f"✅ COMPRA realizada: {quantidade_btc:.6f} {ativo} a {preco_atual:.2f} BRL (Confiança: {confianca_ia:.2f}). Valor alocado: {capital_a_alocar_brl:.2f} BRL.")
        else:
            print("🛑 Saldo em BRL insuficiente ou confiança da IA muito baixa para COMPRAR.")
            transacao = {
                'timestamp': timestamp,
                'ativo': ativo,
                'acao': 'COMPRAR_FALHA',
                'motivo': 'Saldo insuficiente ou baixa confiança',
                'preco_btc': preco_atual,
                'confianca_ia': confianca_ia
            }
            transacoes.append(transacao)

    elif acao == "VENDER":
        quantidade_btc_a_vender = btc_disponivel * (max_pct_btc_a_vender * confianca_ia)
        if quantidade_btc_a_vender > 0.00000001: # Pequeno valor para evitar problemas de float
            valor_brl_venda = quantidade_btc_a_vender * preco_atual
            brl_disponivel += valor_brl_venda
            btc_disponivel -= quantidade_btc_a_vender

            # Cálculo simplificado de lucro/prejuízo
            # Em um sistema real, você precisaria gerenciar o custo médio de aquisição ou FIFO/LIFO
            # Para este exemplo, consideraremos que a venda é baseada no custo médio atual de todo o BTC
            custo_total_btc = sum(t['valor_brl'] for t in transacoes if t['acao'] == 'COMPRAR')
            total_btc_comprado = sum(t['quantidade_btc'] for t in transacoes if t['acao'] == 'COMPRAR')

            lucro_brl = 0
            if total_btc_comprado > 0:
                custo_medio_por_btc = custo_total_btc / total_btc_comprado
                lucro_brl = (preco_atual - custo_medio_por_btc) * quantidade_btc_a_vender

            transacao = {
                'timestamp': timestamp,
                'ativo': ativo,
                'acao': 'VENDER',
                'preco_btc': preco_atual,
                'quantidade_btc': quantidade_btc_a_vender,
                'valor_brl_venda': valor_brl_venda,
                'lucro_brl_estimado': lucro_brl,
                'confianca_ia': confianca_ia
            }
            transacoes.append(transacao)
            print(f"💰 VENDA realizada: {quantidade_btc_a_vender:.6f} {ativo} a {preco_atual:.2f} BRL (Confiança: {confianca_ia:.2f}). Lucro/Prejuízo estimado: {lucro_brl:.2f} BRL.")
        else:
            print("🛑 Nenhum BTC para VENDER ou confiança da IA muito baixa para vender.")
            transacao = {
                'timestamp': timestamp,
                'ativo': ativo,
                'acao': 'VENDER_FALHA',
                'motivo': 'Nenhum BTC para vender ou baixa confiança',
                'preco_btc': preco_atual,
                'confianca_ia': confianca_ia
            }
            transacoes.append(transacao)

    elif acao == "ESPERAR":
        print("⏳ ESPERAR: Nenhuma ação tomada.")
        transacao = {
            'timestamp': timestamp,
            'ativo': ativo,
            'acao': 'ESPERAR',
            'preco_btc': preco_atual,
            'confianca_ia': confianca_ia
        }
        transacoes.append(transacao)

    portfolio_saldo['brl'] = brl_disponivel
    portfolio_saldo['btc'] = btc_disponivel
    portfolio_saldo['transacoes'] = transacoes
    return portfolio_saldo
# ----------------------------------------------------------------------# Código da célulaimport time
import datetime # Adicionado para usar datetime.datetime.now()

# Modificar a inicialização do portfólio
portfolio = {
    'brl': 100.0,  # 100 BRL iniciais
    'btc': 0.0,
    'transacoes': []
}

MAX_CAPITAL_POR_TRANSACAO_BRL = 10.0 # BRL MÁXIMO a ser usado em cada compra, ponderado pela confiança
MAX_PCT_BTC_A_VENDER = 0.5 # Percentual MÁXIMO de BTC a ser vendido, ponderado pela confiança

# Lista para armazenar o histórico do portfólio em cada passo da simulação
portfolio_history = []

print(f"Portfólio inicial (para simulação de 10 operações pequenas): {portfolio['brl']:.2f} BRL, {portfolio['btc']:.6f} BTC")

num_operacoes = 10
profit_target_pct = 2.0 # 2% de lucro

for i in range(num_operacoes):
    print(f"\n--- Simulação de Operação {i+1}/{num_operacoes} ---")

    # 1. Obter dados de mercado atualizados
    try:
        df_candles = pegar_candlesticks("btcbrl", "1m", "10")
        df_ticker = obter_ticker("btcbrl")
        df_orderbook = obter_orderbook("btcbrl")

        if df_candles is None or df_ticker is None or df_orderbook is None:
            print("⚠️ Erro ao obter dados de mercado. Pulando esta iteração.")
            time.sleep(5) # Espera antes de tentar novamente
            # Record current portfolio state even if data fetch fails
            current_timestamp = datetime.datetime.now().isoformat()
            portfolio_history.append({
                'timestamp': current_timestamp,
                'brl_balance': portfolio['brl'],
                'btc_balance': portfolio['btc'],
                'btc_price_at_step': contexto_mercado.get('preco_atual') if 'contexto_mercado' in locals() else 0.0, # Try to get if context exists
                'total_value_brl': portfolio['brl'] + (portfolio['btc'] * (contexto_mercado.get('preco_atual') if 'contexto_mercado' in locals() else 0.0))
            })
            continue

        contexto_mercado = montar_contexto_mercado(
            df_candles,
            df_ticker,
            df_orderbook
        )
    except Exception as e:
        print(f"⚠️ Erro ao montar contexto de mercado: {e}. Pulando esta iteração.")
        time.sleep(5)
        # Record current portfolio state even if context creation fails
        current_timestamp = datetime.datetime.now().isoformat()
        portfolio_history.append({
            'timestamp': current_timestamp,
            'brl_balance': portfolio['brl'],
            'btc_balance': portfolio['btc'],
            'btc_price_at_step': contexto_mercado.get('preco_atual') if 'contexto_mercado' in locals() else 0.0,
            'total_value_brl': portfolio['brl'] + (portfolio['btc'] * (contexto_mercado.get('preco_atual') if 'contexto_mercado' in locals() else 0.0))
        })
        continue

    # 2. Obter decisão da IA
    decisao_str = analisar_mercado_wrapper(contexto_mercado)
    confianca_ia = 0.0 # Valor padrão para confiança
    decisao_ia_dict = {}
    try:
        decisao_ia_dict = json.loads(decisao_str)
        acao_ia = decisao_ia_dict['acao']
        confianca_ia = decisao_ia_dict.get('confianca', 0.0) # Extrai a confiança, padrão 0.0
    except (json.JSONDecodeError, TypeError) as e:
        print(f"❌ Erro ao decodificar JSON da IA ou resposta inválida: {decisao_str}. Erro: {e}. Usando 'ESPERAR'.")
        acao_ia = "ESPERAR"
        confianca_ia = 0.0 # Confiança zero se a resposta for inválida

    acao_final = acao_ia # Começa com a decisão da IA

    # 3. Lógica de VENDA com 2% de lucro
    if portfolio['btc'] > 0: # Se temos BTC no portfólio
        # Calcular custo médio de aquisição a partir das transações de compra
        compras_validas = [t for t in portfolio['transacoes'] if t['acao'] == 'COMPRAR']
        if compras_validas:
            custo_total_btc = sum(t['valor_brl'] for t in compras_validas)
            total_btc_comprado = sum(t['quantidade_btc'] for t in compras_validas)

            if total_btc_comprado > 0:
                custo_medio_por_btc = custo_total_btc / total_btc_comprado
                preco_atual = contexto_mercado.get('preco_atual')

                if preco_atual and preco_atual >= custo_medio_por_btc * (1 + profit_target_pct / 100):
                    acao_final = "VENDER"
                    # Ajusta confiança da IA para 1.0 para forçar a venda total por lucro
                    confianca_ia = 1.0 # Garante que a venda por lucro seja executada com força total
                    print(f"🎯 Condição de LUCRO de {profit_target_pct}% atingida! Forçando ação: VENDER.")
                else:
                    print(f"🔍 Preço atual ({preco_atual:.2f}) não atingiu lucro de {profit_target_pct}% sobre custo médio ({custo_medio_por_btc:.2f}). Mantendo decisão da IA.")
            else:
                print("⚠️ Histórico de compras está vazio, não é possível calcular o custo médio para lucro. Mantendo decisão da IA.")
        else:
            print("⚠️ Histórico de compras está vazio, não é possível calcular o custo médio para lucro. Mantendo decisão da IA.")

    # 4. Gerenciar transação
    print(f"Decisão da IA: {acao_ia} (Confiança: {decisao_ia_dict.get('confianca', 0.0):.2f}) -> Ação Final: {acao_final} (Confiança usada: {confianca_ia:.2f})")
    # Passar a confiança da IA para a função gerenciar_transacoes
    portfolio = gerenciar_transacoes(acao_final, contexto_mercado, portfolio, confianca_ia,
                                     max_capital_por_transacao_brl=MAX_CAPITAL_POR_TRANSACAO_BRL,
                                     max_pct_btc_a_vender=MAX_PCT_BTC_A_VENDER)

    print(f"Portfólio após operação: BRL={portfolio['brl']:.2f}, BTC={portfolio['btc']:.6f}")

    # Capturar o estado do portfólio após a operação
    current_timestamp = datetime.datetime.now().isoformat()
    current_price = contexto_mercado.get('preco_atual', 0.0) # Use 0 if price isn't available

    # Calcular valor total do portfólio em BRL
    total_value_brl = portfolio['brl'] + (portfolio['btc'] * current_price)

    portfolio_history.append({
        'timestamp': current_timestamp,
        'brl_balance': portfolio['brl'],
        'btc_balance': portfolio['btc'],
        'btc_price_at_step': current_price,
        'total_value_brl': total_value_brl
    })

    # Opcional: Adicionar um pequeno delay para simular o tempo entre as operações
    time.sleep(5)

print("\n--- Simulação Concluída ---")
print(f"Portfólio final: BRL={portfolio['brl']:.2f}, BTC={portfolio['btc']:.6f}")
print("Histórico de Transações Detalhado:")
for tx in portfolio['transacoes']:
    print(json.dumps(tx, indent=2))
# ----------------------------------------------------------------------# Código da célulaimport matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Criar um DataFrame a partir do portfolio_history
df_portfolio = pd.DataFrame(portfolio_history)
df_portfolio['timestamp'] = pd.to_datetime(df_portfolio['timestamp'])

# Plotar o valor total do portfólio
plt.figure(figsize=(12, 6))
sns.lineplot(x='timestamp', y='total_value_brl', data=df_portfolio)
plt.title('Evolução do Valor Total do Portfólio (BRL)')
plt.xlabel('Tempo')
plt.ylabel('Valor Total em BRL')
plt.grid(True)
plt.tight_layout()
plt.show()

# Opcional: Plotar saldos de BRL e BTC separadamente para análise mais detalhada
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
# ----------------------------------------------------------------------# Código da célula
