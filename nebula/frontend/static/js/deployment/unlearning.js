// Unlearning Module
const UnlearningManager = (function() {
    const UNLEARNING_METHODS = {
        BASIC_RETRAINING: 'Basic Retraining',
        KNOWLEDGE_DISTILLATION: 'Knowledge Distillation',
        GRADIENT_ASCENT: 'Gradient Ascent',
    };

    const Config = {
        method: UNLEARNING_METHODS.BASIC_RETRAINING,
        unlearningNodePercent: 0,
        unlearningRound: 5,
    };

    function initializeEventListeners() {
        const unlearningMethod = document.getElementById('unlearningMethod');
        const unlearningNodePercentInput = document.getElementById('unlearningNodePercentInput');
        const unlearningNodePercentValue = document.getElementById('unlearningNodePercentValue');
        const unlearningRound = document.getElementById('unlearningRound');
        const rounds = document.getElementById('rounds');

        unlearningMethod.addEventListener('change', function() {
            Config.method = unlearningMethod.value;
        });

        unlearningNodePercentInput.addEventListener('change', function() {
            unlearningNodePercentValue.value = unlearningNodePercentInput.value;
            Config.unlearningNodePercent = unlearningNodePercentInput.value;
        });

        unlearningNodePercentValue.addEventListener('change', function() {
            unlearningNodePercentInput.value = unlearningNodePercentValue.value;
            Config.unlearningNodePercent = unlearningNodePercentValue.value;
        });

        unlearningRound.addEventListener('change', function() {
            unlearningRound.value = Math.min(Number(unlearningRound.value), Number(rounds.value) - 1);
            Config.unlearningRound = unlearningRound.value;
        });

        rounds.addEventListener('change', function() {
            unlearningRound.value = Math.max(0, Math.min(Number(unlearningRound.value), Number(rounds.value) - 1));
            Config.unlearningRound = unlearningRound.value;
        });
    }

    function resetUnlearningConfig() {
        Config.method = UNLEARNING_METHODS.BASIC_RETRAINING;
        document.getElementById("unlearningMethod").value = Config.method;

        Config.unlearningNodePercent = 0;
        document.getElementById("unlearningNodePercentInput").value = Config.unlearningNodePercent;
        document.getElementById("unlearningNodePercentValue").value = Config.unlearningNodePercent;

        Config.unlearningRound = 5;
        document.getElementById("unlearningRound").value = Config.unlearningRound;
    }

    return {
        UNLEARNING_METHODS,
        initializeEventListeners,
        getUnlearningConfig: () => Config,
        resetUnlearningConfig
    };
})();

export default UnlearningManager;
