/* Keep delayed trace responses from replacing a newer metrics selection. */
(function () {
  "use strict";

  function create() {
    let generation = 0;
    return {
      begin() {
        generation += 1;
        return generation;
      },
      isCurrent(ticket) {
        return ticket === generation;
      },
      async apply(ticket, request, onValue, onError) {
        try {
          const value = await request();
          if (ticket === generation) onValue(value);
        } catch (error) {
          if (ticket === generation && onError) onError(error);
        }
      },
    };
  }

  window.TraceSelection = { create };
})();
