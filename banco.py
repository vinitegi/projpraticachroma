from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# -----------------------------------------------------------------------------
# Base de Dados: 10 Documentos de Suporte de TI
# Estrutura: (id, texto do documento, metadados)
# -----------------------------------------------------------------------------
DOCUMENTOS = [
    (
        "ti-01",
        "Para redefinir sua senha, acesse a opção Esqueci minha senha no portal de login.",
        {"categoria": "acesso", "tipo": "senha"},
    ),
    (
        "ti-02",
        "A rede sem fio corporativa está disponível em todos os blocos e laboratórios.",
        {"categoria": "rede", "tipo": "wifi"},
    ),
    (
        "ti-03",
        "Caso a impressora esteja sem papel ou com toner fraco, abra um chamado no helpdesk.",
        {"categoria": "hardware", "tipo": "impressora"},
    ),
    (
        "ti-04",
        "Para solicitar a troca ou substituição de monitor quebrado, contate a equipe de TI.",
        {"categoria": "hardware", "tipo": "perifericos"},
    ),
    (
        "ti-05",
        "Solicitações de novos softwares devem ser feitas com aprovação previa do gestor.",
        {"categoria": "software", "tipo": "instalacao"},
    ),
    (
        "ti-06",
        "A VPN deve ser utilizada obrigatoriamente para acessos remotos fora da empresa.",
        {"categoria": "acesso", "tipo": "vpn"},
    ),
    (
        "ti-07",
        "Computadores travando ou muito lentos devem ser reiniciados antes de abrir um chamado.",
        {"categoria": "hardware", "tipo": "manutencao"},
    ),
    (
        "ti-08",
        "E-mails suspeitos com links desconhecidos devem ser reportados ao time de segurança.",
        {"categoria": "seguranca", "tipo": "phishing"},
    ),
    (
        "ti-09",
        "O backup dos ficheiros do seu computador é realizado automaticamente todas as noites.",
        {"categoria": "seguranca", "tipo": "backup"},
    ),
    (
        "ti-10",
        "Para cadastrar um novo funcionário no sistema de e-mail, envie o formulário do RH.",
        {"categoria": "acesso", "tipo": "cadastro"},
    ),
]


def main() -> None:
    # Modelo de Embeddings
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Inicialização do banco persistente e criação/obtenção da collection
    client = PersistentClient(path="chroma_db_ti")
    collection = client.get_or_create_collection(
        name="suporte_ti",
        embedding_function=embedding_fn,
    )

    # 1. Inserir os 10 documentos
    collection.upsert(
        ids=[doc[0] for doc in DOCUMENTOS],
        documents=[doc[1] for doc in DOCUMENTOS],
        metadatas=[doc[2] for doc in DOCUMENTOS],
    )

    # 2. Mostrar contagem total de documentos
    print("=" * 60)
    print(f"1. Total de documentos indexados: {collection.count()}")
    print("=" * 60)

    # 3. Demonstrar dimensões do embedding
    exemplo_doc = collection.get(ids=["ti-01"], include=["embeddings"])
    if exemplo_doc["embeddings"] is not None:
        vetor = exemplo_doc["embeddings"][0]
        print(f"\n2. Dimensões do embedding gerado: {len(vetor)} dimensões\n")

    # 4 e 5. Consultas semânticas (3 perguntas -> Top 3 resultados cada)
    perguntas = [
        "Como me conectar à internet sem fio da empresa?",
        "O que fazer se o meu ecrã ou monitor avariar?",
        "Não consigo entrar no sistema porque esqueci a minha credencial de acesso.",
    ]

    print("=" * 60)
    print("3. Consultas Semânticas (Top-3 por Pergunta)")
    print("=" * 60)

    for idx, pergunta in enumerate(perguntas, 1):
        print(f"\n--- Pergunta {idx}: '{pergunta}' ---")
        resultado = collection.query(
            query_texts=[pergunta],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        for doc_id, texto, metadata, distancia in zip(
            resultado["ids"][0],
            resultado["documents"][0],
            resultado["metadatas"][0],
            resultado["distances"][0],
        ):
            print(f" • ID: {doc_id}")
            print(f" Documento: {texto}")
            print(f" Metadata: {metadata}")
            print(f" Distância: {distancia:.4f}")

    # 6. Busca filtrada com `where` por categoria
    print("\n" + "=" * 60)
    print("4. Busca com Filtro por Metadados (categoria = 'acesso')")
    print("=" * 60)

    pergunta_filtro = "Como ter acesso aos sistemas corporativos?"
    resultado_filtrado = collection.query(
        query_texts=[pergunta_filtro],
        n_results=3,
        where={"categoria": "acesso"},
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nPergunta com filtro: '{pergunta_filtro}'")
    for doc_id, texto, metadata, distancia in zip(
        resultado_filtrado["ids"][0],
        resultado_filtrado["documents"][0],
        resultado_filtrado["metadatas"][0],
        resultado_filtrado["distances"][0],
    ):
        print(f"• ID: {doc_id}")
        print(f" Documento: {texto}")
        print(f" Metadata: {metadata}")
        print(f" Distância: {distancia:.4f}")


if __name__ == "__main__":
    main()