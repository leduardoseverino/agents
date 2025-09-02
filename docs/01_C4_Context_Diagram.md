# C4 Context Diagram

## 🎯 Visão Contextual do Sistema

### Sistema Principal
O sistema principal é um repositório de código que contém principalmente arquivos Python e alguns arquivos JavaScript. Ele parece ser uma coleção de ferramentas e scripts relacionados a agentes conversacionais, avaliação de agentes, e integrações com serviços externos como LlamaIndex.

### Usuários e Atores
- **Desenvolvedores**: Pessoas que contribuem para o código-fonte do projeto.
- **Usuários Finais**: Pessoas que utilizam as ferramentas e scripts fornecidos pelo sistema para suas próprias necessidades.

### Sistemas Externos
- **LlamaIndex**: Serviço externo utilizado para motores de consulta e agentes conversacionais.
- **WebRTC e WebSocket**: Tecnologias utilizadas para comunicação em tempo real nos notebooks de agentchat.

### Interações Principais
O sistema se comunica com o mundo externo principalmente através de APIs e serviços como LlamaIndex, além de utilizar tecnologias de comunicação em tempo real como WebRTC e WebSocket.

## 📊 Diagrama de Contexto C4

```mermaid
C4Context
    title Diagrama de Contexto - AG2

    Person(developer, "Desenvolvedor", "Contribui para o código-fonte")
    Person(user, "Usuário Final", "Utiliza as ferramentas e scripts")

    System(system, "AG2", "Repositório de código com ferramentas e scripts para agentes conversacionais")

    System_Ext(llamaindex, "LlamaIndex", "Serviço externo para motores de consulta")
    System_Ext(webrtc, "WebRTC", "Tecnologia para comunicação em tempo real")
    System_Ext(websocket, "WebSocket", "Tecnologia para comunicação em tempo real")

    Rel(developer, system, "Desenvolve e mantém")
    Rel(user, system, "Utiliza ferramentas e scripts")
    Rel(system, llamaindex, "Consome API")
    Rel(system, webrtc, "Utiliza")
    Rel(system, websocket, "Utiliza")
```

## 🔗 Integrações Identificadas
- **LlamaIndex**: Utilizado para motores de consulta e agentes conversacionais.
- **WebRTC**: Utilizado para comunicação em tempo real nos notebooks de agentchat.
- **WebSocket**: Utilizado para comunicação em tempo real nos notebooks de agentchat.