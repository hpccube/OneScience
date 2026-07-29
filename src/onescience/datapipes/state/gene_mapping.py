"""Gene-symbol mapping with an explicit opt-in for remote services."""

import logging
import time

import numpy as np

# onescience: explicit-online-feature
GENE_NAME_ENSEMBL_MAP = {"GATD3A": "ENSMUSG00000053329", "GATD3B": "ENSG00000160221"}


def convert_gene_symbols_to_ensembl_rest(gene_symbols, species="human", *, allow_online=False):
    """Resolve symbols through Ensembl when the caller explicitly permits networking."""
    if not allow_online:
        raise RuntimeError("Ensembl REST access is disabled; pass allow_online=True to opt in")

    import requests

    server = "https://grch37.rest.ensembl.org"
    species_map = {"human": "homo_sapiens", "mouse": "mus_musculus", "rat": "rattus_norvegicus"}
    species_name = species_map.get(species.lower(), species)
    gene_to_ensembl = {}

    for symbol in gene_symbols:
        ext = f"/lookup/symbol/{species_name}/{symbol}?"
        response = requests.get(server + ext, headers={"Content-Type": "application/json"}, timeout=30)
        if response.status_code != 200:
            logging.warning("Failed to retrieve data for %s: %s", symbol, response.status_code)
            continue
        decoded = response.json()
        if "id" in decoded:
            gene_to_ensembl[symbol] = decoded["id"]
        time.sleep(0.1)
    return gene_to_ensembl


def convert_symbols_to_ensembl(adata, *, allow_online=False):
    """Resolve AnnData symbols through MyGene and Ensembl with explicit opt-in."""
    if not allow_online:
        raise RuntimeError("Remote gene mapping is disabled; pass allow_online=True to opt in")

    import mygene

    gene_symbols = adata.var_names.tolist()
    results = mygene.MyGeneInfo().querymany(
        gene_symbols,
        scopes="symbol",
        fields="ensembl.gene",
        species="human",
    )
    symbol_to_ensembl = {}
    for result in results:
        if "ensembl" in result and not result.get("notfound", False):
            if isinstance(result["ensembl"], list):
                symbol_to_ensembl[result["query"]] = result["ensembl"][0]["gene"]
            else:
                symbol_to_ensembl[result["query"]] = result["ensembl"]["gene"]

    for symbol in gene_symbols:
        if symbol_to_ensembl.get(symbol) is None:
            resolved = convert_gene_symbols_to_ensembl_rest([symbol], allow_online=True)
            if resolved:
                symbol_to_ensembl[symbol] = resolved[symbol]
                logging.info("Converted %s to %s using REST API", symbol, resolved[symbol])

    for symbol in gene_symbols:
        if symbol_to_ensembl.get(symbol) is None and symbol in GENE_NAME_ENSEMBL_MAP:
            symbol_to_ensembl[symbol] = GENE_NAME_ENSEMBL_MAP[symbol]
    symbol_to_ensembl["PBK"] = "ENSG00000168078"
    return [symbol_to_ensembl.get(symbol, np.nan) for symbol in gene_symbols]
