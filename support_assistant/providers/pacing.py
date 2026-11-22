from collections.abc import Callable

from support_assistant.providers.base import Provider, ProviderRequest, ProviderResponse


# provedor que espera o intervalo entre chamadas, inclusive antes de um reparo
class SpacedProvider(Provider):
    # envolve o provedor real com o intervalo e a função de espera
    def __init__(
        self, provider: Provider, interval_s: float, sleep: Callable[[float], None]
    ) -> None:
        self._provider = provider
        self._interval_s = interval_s
        self._sleep = sleep
        self._called = False

    # nome do provedor real, gravado em cada execução
    @property
    def name(self) -> str:
        return self._provider.name

    # espera o intervalo a partir da segunda chamada e repassa a requisição
    def complete(self, request: ProviderRequest) -> ProviderResponse:
        if self._called and self._interval_s > 0:
            self._sleep(self._interval_s)
        self._called = True
        return self._provider.complete(request)
